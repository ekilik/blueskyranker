import polars as pl
import json
import re
import unicodedata
import os
import argparse
from typing import List, Dict, Optional
import stanza
from tqdm import tqdm


class ActorEnricher:
    """
    Enriches actor annotations from LLM output by:
    1. Parsing and cleaning LLM JSON output
    2. Expanding actors to row-level data
    3. Calculating actor statistics per function category
    4. Extracting core names using NER
    5. Enriching political actors with party affiliations from multiple sources
    """

    def __init__(
            self,
            actor_data_path: Optional[str] = None,
            actor_df: Optional[pl.DataFrame] = None,
            id_column: str = 'uri',
            language: str = 'en',
            center_parties: bool = True,
            politicians_data_path: Optional[str] = None):

        """
        Initialize ActorEnricher.

        Args:
            actor_data_path: Path to CSV file with actor annotations, optional
            actor_df: DataFrame with actor annotations (alternative to path if path not provided)
            id_column: Name of the column containing unique article identifiers
            language: Language for NER processing ('en', 'nl', etc.)
            center_parties: Whether the country has center parties in addition to left and right
            politicians_data_path: Path to CSV with party reference data (columns: name, party, lrgen_category)
        """

        self.actor_data_path = actor_data_path
        self.id_column = id_column
        self.language = language
        self.center_parties = center_parties

        # Load actor data
        self.actor_df = actor_df if actor_df is not None else self._load_actor_data()
        # Initialize NLP pipeline for NER
        print(f"Initializing Stanza NLP pipeline for language: {language}")
        self.nlp = stanza.Pipeline(
            language=self.language,
            processors='tokenize,ner',
            tokenize_no_ssplit=True,
            verbose=False
        )

        # Load reference data as lists of dicts for efficient iteration
        _political_df = self._load_politicians_data(politicians_data_path)
        if _political_df is not None:
            self.politician_reference_df: Optional[List[Dict]] = (
                _political_df.select(['name', 'party', 'lrgen_category'])
                .unique().to_dicts()
            )
            self.ideology_reference_df: Optional[List[Dict]] = (
                _political_df.select(['party', 'lrgen_category'])
                .unique().to_dicts()
            )
        else:
            self.politician_reference_df = None
            self.ideology_reference_df = None

        # add two parties to ideology reference data if not already present: GROENLINKS-PVDA left, and CU center
        if self.language == 'nl' and self.ideology_reference_df is not None:
            additional_parties = [
                {'party': 'GROENLINKS-PVDA', 'lrgen_category': 'left'},
                {'party': 'GROENLINKS', 'lrgen_category': 'left'},
                {'party': 'PVDA', 'lrgen_category': 'left'},
                {'party': 'CU', 'lrgen_category': 'center'},
            ]
            existing_parties = {row['party'] for row in self.ideology_reference_df}
            for ap in additional_parties:
                if ap['party'] not in existing_parties:
                    self.ideology_reference_df.append(ap)

    def _load_actor_data(self) -> pl.DataFrame:
            """Load actor data from CSV file into a DataFrame"""
            if self.actor_data_path:
                return pl.read_csv(self.actor_data_path, separator=';')
            else:
                raise ValueError("No actor data path provided and no DataFrame was passed.")

    def _load_politicians_data(self, path: Optional[str]) -> Optional[pl.DataFrame]:
        """Load party reference data for matching."""
        if path and os.path.exists(path):
            print(f"Loading party reference data from {path}")
            df = pl.read_csv(path, separator=';')
            return df.unique(subset=['name'], keep='first')
        return None

    def _parse_actors_json(self, actors_json_str):
        """Parse the JSON string and extract actor lists"""
        if actors_json_str is None:
            return [], [], []
        cleaned = re.sub(r"\s+", " ", actors_json_str)
        cleaned = re.sub(r"^```(?:json)?|```$", "",
                         cleaned,
                         flags=re.IGNORECASE | re.MULTILINE
                         ).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        try:
            data = json.loads(cleaned)
            actors = data.get('actors', [])
            names = [actor.get('actor_name', '') for actor in actors]
            functions = [actor.get('actor_function', '') for actor in actors]
            parties = [actor.get('actor_pp', '') for actor in actors]
            return names, functions, parties
        except (json.JSONDecodeError, AttributeError):
            return [], [], []

    def expand_actors_to_rows(self, actor_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        Transform DataFrame from article-level to actor-level.
        Each actor becomes a separate row with article metadata.
        Articles without actors get a row with empty values in actor columns.

        Args:
            actor_df: Optional DataFrame to expand. If None, uses self.actor_df

        Returns:
            DataFrame with one row per actor
        """
        if actor_df is None:
            actor_df = self.actor_df

        if actor_df.is_empty():
            return pl.DataFrame()

        # Validate that the ID column exists
        if self.id_column not in actor_df.columns:
            raise ValueError(
                f"ID column '{self.id_column}' not found in DataFrame. "
                f"Available columns: {list(actor_df.columns)}")

        rows = []
        for row in actor_df.to_dicts():
            names, functions, parties = self._parse_actors_json(row['news_actors'])
            raw_output = row.get('news_actors_raw', '')

            # If no actors found, create one row with empty values
            if len(names) == 0:
                rows.append({
                    self.id_column: row[self.id_column],
                    'actor_name': '',
                    'actor_function': '',
                    'actor_pp': '',
                    'news_actors_raw': raw_output,
                })
            else:
                # Each actor as a separate row
                for i in range(len(names)):
                    rows.append({
                        self.id_column: row[self.id_column],
                        'actor_name': names[i] if i < len(names) else '',
                        'actor_function': functions[i] if i < len(functions) else '',
                        'actor_pp': parties[i] if i < len(parties) else '',
                        'news_actors_raw': raw_output,
                    })

        if not rows:
            return pl.DataFrame({
                self.id_column: pl.Series([], dtype=pl.Utf8),
                'actor_name': pl.Series([], dtype=pl.Utf8),
                'actor_function': pl.Series([], dtype=pl.Utf8),
                'actor_pp': pl.Series([], dtype=pl.Utf8),
                'news_actors_raw': pl.Series([], dtype=pl.Utf8),
            })
        return pl.DataFrame(rows)

    # calculate nr of unique actors per function per article
    def calculate_actors_per_function(self, actor_df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate number of unique actors per function for each article.

        Args:
            actor_df: Optional actor-level DataFrame. If None, expands self.actor_df

        Returns:
            DataFrame with columns: id_column, nr_actors_a, nr_actors_b, nr_actors_c,
            nr_actors_d, nr_actors_total
        """
        if actor_df is None:
            actor_df = self.expand_actors_to_rows()

        if actor_df.is_empty():
            print("Actor DataFrame is empty. Returning empty DataFrame.")
            return pl.DataFrame()

        # Keep only valid functions
        valid_functions = ['a', 'b', 'c', 'd']
        actor_df = actor_df.filter(pl.col('actor_function').is_in(valid_functions))

        if actor_df.is_empty():
            print("No actors with valid functions (a, b, c, d) found.")
            return pl.DataFrame()

        # Count unique actor names per (id, function)
        unique_counts = (
            actor_df
            .group_by([self.id_column, 'actor_function'])
            .agg(pl.col('actor_name').n_unique().alias('nr_actors'))
        )

        # Pivot to have functions as columns
        functions_df = unique_counts.pivot(
            index=self.id_column,
            on='actor_function',
            values='nr_actors',
        ).fill_null(0)

        # if one of the function columns is missing, add it with zeros
        for func in valid_functions:
            if func not in functions_df.columns:
                functions_df = functions_df.with_columns(pl.lit(0).alias(func))

        # Rename columns
        rename_map = {f: f'nr_actors_{f}' for f in valid_functions if f in functions_df.columns}
        functions_df = functions_df.rename(rename_map)

        # Calculate total number of unique actors
        actor_cols = [c for c in functions_df.columns if c.startswith('nr_actors_')]
        functions_df = functions_df.with_columns(
            pl.sum_horizontal([pl.col(c) for c in actor_cols]).alias('nr_actors_total')
        )

        return functions_df

    def _clean_actor_name(self, name: str) -> str:
        """Remove text in parentheses and extra whitespace."""
        if name is None:
            return ""
        return re.sub(r"\(.*?\)", "", str(name)).strip()

    def extract_core_name(self, full_name: str) -> Optional[str]:
        """
        Extract the core person name using NER.

        Args:
            full_name: Full actor name string

        Returns:
            Extracted person name or None if not a person
        """
        if full_name is None or not str(full_name).strip():
            return None

        clean_name = self._clean_actor_name(full_name)
        if not clean_name:
            return None

        doc = self.nlp(clean_name)

        # Get unique person entity names
        person_entities = list({
            ent.text for ent in doc.ents
            if ent.type in ['PER', 'PERSON']
        })

        # Return first person entity found
        if person_entities:
            entity_str = person_entities[0].strip()
            return entity_str.title() if entity_str else None

        return None

    def _query_sparql(self, sparql: str) -> Dict:
        """Execute SPARQL query against Wikidata."""
        WDQS = "https://query.wikidata.org/sparql"
        HEADERS = {"User-Agent": "ActorEnricher/1.0"}

        try:
            from SPARQLWrapper import SPARQLWrapper, JSON
        except ImportError as e:
            raise RuntimeError(
                "SPARQLWrapper not installed. Install with: pip install SPARQLWrapper"
            ) from e

        sparqlw = SPARQLWrapper(WDQS, agent=HEADERS["User-Agent"])
        sparqlw.setQuery(sparql)
        sparqlw.setReturnFormat(JSON)

        return sparqlw.query().convert()

    def _search_wikidata(self, name: str, language: str) -> Optional[str]:
        """Search for a person on Wikidata and return their QID."""
        HEADERS = {"User-Agent": "ActorEnricher/1.0"}

        try:
            import requests
        except ImportError as e:
            raise RuntimeError(
                "requests not installed. Install with: pip install requests"
            ) from e

        params = {
            "action": "wbsearchentities",
            "search": name,
            "language": language,
            "format": "json",
            "limit": 1
        }

        resp = requests.get(
            "https://www.wikidata.org/w/api.php",
            params=params,
            headers=HEADERS,
            timeout=10
        )
        resp.raise_for_status()
        hits = resp.json().get("search", [])

        return hits[0]["id"] if hits else None

    def get_latest_party_from_wikidata(self, name: str, language: str
                                       ) -> Optional[Dict[str, Optional[str]]]:
        """
        Query Wikidata for a person's latest party affiliation.

        Args:
            name: Person's name
            language: Language code for labels

        Returns:
            Dictionary with 'party_name' and 'party_name_short' keys, or None if not found
        """
        qid = self._search_wikidata(name, language=language)
        if not qid:
            return None

        sparql = f"""
        SELECT ?partyLabel ?shortName ?start ?end WHERE {{
            VALUES ?person {{ wd:{qid} }}
            ?person p:P102 ?stmt .
            ?stmt ps:P102 ?party .
            OPTIONAL {{ ?stmt pq:P580 ?start. }}
            OPTIONAL {{ ?stmt pq:P582 ?end. }}
            OPTIONAL {{ ?party wdt:P1813 ?shortName. }}
            SERVICE wikibase:label {{
                bd:serviceParam wikibase:language "{language},en".
            }}
        }}
        """

        try:
            results = self._query_sparql(sparql)
        except Exception as e:
            print(f"SPARQL query failed for {name}: {e}")
            return None

        records = [{
            "party": r["partyLabel"]["value"],
            "short_name": r.get("shortName", {}).get("value"),
            "start": r.get("start", {}).get("value"),
            "end": r.get("end", {}).get("value"),
        } for r in results["results"]["bindings"]]

        if not records:
            return None

        # Sort by start/end descending (ISO date strings sort lexicographically)
        records.sort(key=lambda x: (x['start'] or '', x['end'] or ''), reverse=True)
        latest = records[0]
        return {
            "party_name": latest["party"],
            "party_name_short": latest["short_name"] or None,
        }

    def fetch_party_info(self, name: str, language: str = "en") -> Dict[str, Optional[str]]:
        """
        Wrapper to safely fetch party information from Wikidata.

        Args:
            name: Person's name
            language: Language code

        Returns:
            Dict with party_name and party_name_short
        """
        try:
            result = self.get_latest_party_from_wikidata(name, language=language)
            if result:
                return {
                    "party_name": result["party_name"],
                    "party_name_short": result["party_name_short"],
                }
        except Exception as e:
            print(f"Error fetching party info for {name}: {e}")

        return {"party_name": None, "party_name_short": None}

    def _normalize_string(self, s: str) -> str:
        """Unicode-safe normalization + uppercase."""
        if not isinstance(s, str):
            return ""
        s = unicodedata.normalize("NFKC", s)
        return s.strip().upper()

    def _tokenize_name(self, name: str, stop_tokens: set) -> list[str]:
        """
        Normalize + split name into meaningful tokens.
        Removes titles but keeps surname particles.
        """
        name = self._normalize_string(name)

        # Replace hyphens with spaces for flexible matching
        name = name.replace("-", " ")

        tokens = name.split()

        # Remove stop tokens
        tokens = [t for t in tokens if t not in stop_tokens]

        return tokens

    def _extract_surname(self, tokens: list[str]) -> str:
        """
        Extract surname including particles.
        Example:
        ['MARK', 'RUTTE'] → RUTTE
        ['JAN', 'VAN', 'DIJK'] → VAN DIJK
        """
        if not tokens:
            return ""

        surname_parts = []
        i = len(tokens) - 1

        # Always include last token
        surname_parts.insert(0, tokens[i])
        i -= 1

        # Include preceding particles
        while i >= 0 and tokens[i] in {"VAN", "DER", "DEN", "DE", "VON", "ZU", "TEN", "TER", "LA", "LE", "DI"}:
            surname_parts.insert(0, tokens[i])
            i -= 1

        return " ".join(surname_parts)

    def _surname_match(self, partial_name: str, full_name: str) -> bool:
        """
        Precision-controlled surname matching for political actors.
        """

        if not partial_name or not full_name:
            return False

                # First we make a list of possible function titles that could precede surnames in ENG, NL, DE, TR
        STOP_TOKENS = {
            # NL
            "KAMERLID", "TWEEDE", "EERSTE", "LID", "STAATSSECRETARIS", "MINISTER",
            "MINISTER-PRESIDENT", "VICEPREMIER", "PREMIER", "FRACTIEVOORZITTER",
            "BURGEMEESTER", "WETHOUDER", "RAADSLID", "SENATOR", "VOORZITTER", "DEMISSIONAIR",

            # DE
            "BUNDESKANZLER", "KANZLER", "MINISTERPRÄSIDENT", "ABGEORDNETER",
            "ABGEORDNETE", "BUNDESTAG", "LANDTAG", "BÜRGERMEISTER",
            "VORSITZENDER", "GESCHÄFTSFÜHREND",

            # EN
            "PRIME", "MINISTER", "SECRETARY", "STATE", "MP", "MEP",
            "MAYOR", "GOVERNOR", "PRESIDENT", "CHAIR", "CHAIRMAN",
            "CHAIRWOMAN", "SPEAKER", "MEMBER",

            # IE
            "TAOISEACH", "TÁNAISTE", "TD",

            # TR
            "BAKAN", "BAKANI", "BAŞBAKAN", "CUMHURBAŞKANI",
            "MILLETVEKILI", "BELEDIYE", "BELEDIYE BAŞKANI"
        }

        partial_tokens = self._tokenize_name(partial_name, STOP_TOKENS)
        full_tokens = self._tokenize_name(full_name, STOP_TOKENS)

        if not partial_tokens or not full_tokens:
            return False

        partial_surname = self._extract_surname(partial_tokens)
        full_surname = self._extract_surname(full_tokens)

        # --- Step 1: surnames must match exactly ---
        if partial_surname != full_surname:
            return False

        # --- Step 2: If only surname provided ---
        if len(partial_tokens) == 1:
            # avoid false positives like "LI", "NG"
            if len(partial_surname) < 4:
                return False
            return True

        # --- Step 3: If firstname present, require at least one match ---
        partial_firstnames = set(partial_tokens[:-1])
        full_firstnames = set(full_tokens[:-1])

        if partial_firstnames & full_firstnames:
            return True

        return False


    def enrich_political_actors(self,
                                actor_df: Optional[pl.DataFrame] = None,
                                use_wikidata: bool = True,
                                language: str = "en") -> pl.DataFrame:
        """
        Enrichment pipeline for politicians (function 'a', NER person):
        1. Check for party name mentions in actor names (fast, direct signal)
        2. Extract core names using NER for unmatched rows
        3. Match against party reference data (exact → token → original name)
        4. Query Wikidata for still-missing information (if use_wikidata=True)
        5. Merge with ideology scores

        Args:
            actor_df: Optional actor-level DataFrame. If None, expands self.actor_df
            use_wikidata: Whether to query Wikidata for missing party info
            language: Language code for Wikidata queries
        Returns:
            DataFrame with enriched actor information including party affiliations
        """
        if actor_df is None:
            actor_df = self.expand_actors_to_rows()

        if actor_df.is_empty():
            print("Actor DataFrame is empty. Returning empty DataFrame.")
            return pl.DataFrame()

        # Filter for political actors only (function 'a'), work as list of dicts
        political_actors: List[Dict] = actor_df.filter(pl.col('actor_function') == 'a').to_dicts()

        if not political_actors:
            print("No political actors (function 'a') found.")
            return pl.DataFrame()

        print(f"Processing {len(political_actors)} political actor records...")

        # Normalize actor names and initialize all extra columns on every row
        for row in political_actors:
            row['actor_name_upper'] = self._normalize_string(row['actor_name'])
            row['party'] = None
            row['lrgen_category'] = None
            row['matched_name'] = None
            row['core_actor_name'] = None
            row['core_actor_name_upper'] = None

        if self.politician_reference_df is not None:
            for ref_row in self.politician_reference_df:
                ref_row['name'] = self._normalize_string(ref_row['name'])

        # Step 1: Check for party name mentions in actor_name (before NER)
        print("Step 1: Checking for party name mentions in actor names...")
        party_mention_count = 0
        extra_rows: List[Dict] = []

        if self.ideology_reference_df is not None:
            for row in political_actors:
                actor_name = re.sub(r"[-–—/]", " ", row['actor_name_upper'])
                actor_name = re.sub(r"\s+", " ", actor_name).strip()

                matched_parties = []

                for ref_row in self.ideology_reference_df:
                    party_name_original = str(ref_row['party']).strip()
                    party_name_upper = self._normalize_string(party_name_original)
                    party_name_upper = re.sub(r"[-–—/]", " ", party_name_upper)
                    party_name_upper = re.sub(r"\s+", " ", party_name_upper).strip()

                    pattern = rf"(?<!\w){re.escape(party_name_upper)}(?!\w)"

                    if re.search(pattern, actor_name):
                        matched_parties.append({
                            'party': party_name_original,
                            'matched_name': party_name_original,
                            'lrgen_category': ref_row['lrgen_category']
                        })

                # For each matched party, create or update rows
                if matched_parties:
                    party_mention_count += 1
                    row['party'] = matched_parties[0]['party']
                    row['lrgen_category'] = matched_parties[0]['lrgen_category']
                    row['matched_name'] = matched_parties[0]['matched_name']
                    for match in matched_parties[1:]:
                        new_row = dict(row)
                        new_row['party'] = match['party']
                        new_row['lrgen_category'] = match['lrgen_category']
                        new_row['matched_name'] = match['matched_name']
                        extra_rows.append(new_row)

            # Append new rows for additional matches
            political_actors.extend(extra_rows)

        print(f". → Matched {party_mention_count} actors with party name mentions")

        unmatched = [row for row in political_actors if row['party'] is None]
        print(f"Step 2: Extracting core names using NER for {len(unmatched)} actors with no party mentions...")

        for row in tqdm(unmatched, desc="Extracting names"):
            row['core_actor_name'] = self.extract_core_name(row['actor_name'])

        for row in political_actors:
            cn = row['core_actor_name']
            row['core_actor_name_upper'] = self._normalize_string(cn) if cn is not None else None

        if self.politician_reference_df is not None:
            print("Step 2.1: Exact matching on core_actor_name...")
            step1_matched = sum(1 for row in political_actors if row['party'] is not None)

            # Build lookup dict for O(1) exact matching
            exact_lookup = {ref['name']: ref for ref in self.politician_reference_df}

            for row in political_actors:
                if row['party'] is None and row['core_actor_name_upper']:
                    match = exact_lookup.get(row['core_actor_name_upper'])
                    if match:
                        row['party'] = match['party']
                        row['lrgen_category'] = match['lrgen_category']
                        row['matched_name'] = match['name']

            exact_count = sum(1 for row in political_actors if row['party'] is not None) - step1_matched
            print(f"  → Matched {exact_count} actors with exact core_actor_name match")
            seen: set = set()
            unique_matched = []
            for r in political_actors:
                if r['party'] is not None and r['core_actor_name'] not in seen:
                    seen.add(r['core_actor_name'])
                    unique_matched.append({'core_actor_name': r['core_actor_name'], 'matched_name': r['matched_name']})
            print(f"These names are matched with reference data: {unique_matched}")

        # Step 2.2: Token match on core_actor_name (for unmatched rows)
        if self.politician_reference_df is not None:
            print("Step 2.2: Token matching on core_actor_name...")
            token_match_count = 0

            for row in political_actors:
                if row['party'] is None:
                    for ref_row in self.politician_reference_df:
                        if self._surname_match(row['core_actor_name_upper'], ref_row['name']):
                            row['party'] = ref_row['party']
                            row['lrgen_category'] = ref_row['lrgen_category']
                            row['matched_name'] = ref_row['name']
                            token_match_count += 1
                            break

            print(f"→ Matched {token_match_count} actors with token match on core_actor_name")
            seen2: set = set()
            unique_matched2 = []
            for r in political_actors:
                if r['party'] is not None and r['core_actor_name'] not in seen2:
                    seen2.add(r['core_actor_name'])
                    unique_matched2.append({'core_actor_name': r['core_actor_name'], 'matched_name': r['matched_name']})
            print(f"These names are matched with reference data: {unique_matched2}")

            # Step 2.3: Exact + token match on actor_name (for still unmatched rows)
            print("Step 2.3: Matching on original actor_name...")
            actor_name_count = 0

            for row in political_actors:
                if row['party'] is None:
                    actor_name = row['actor_name_upper']
                    match = exact_lookup.get(actor_name)
                    if match:
                        row['party'] = match['party']
                        row['lrgen_category'] = match['lrgen_category']
                        row['matched_name'] = match['name']
                        actor_name_count += 1
                        continue

                    for ref_row in self.politician_reference_df:
                        if self._surname_match(actor_name, ref_row['name']):
                            row['party'] = ref_row['party']
                            row['lrgen_category'] = ref_row['lrgen_category']
                            row['matched_name'] = ref_row['name']
                            actor_name_count += 1
                            break

            print(f"→ Matched {actor_name_count} actors with actor_name matching")

            total_matched = sum(1 for row in political_actors if row['party'] is not None)
            print(f"\nTotal matched: {total_matched} actors with reference data")

        # Step 3: Query Wikidata for missing information
        if use_wikidata:
            missing = [row for row in political_actors if row['party'] is None]

            if missing:
                print(f"Querying Wikidata for {len(missing)} actors with missing party info...")

                unique_names = list({row['core_actor_name'] for row in missing if row['core_actor_name'] is not None})
                print(f"Querying {len(unique_names)} unique names...")

                wikidata_results = {}
                import time
                for name in tqdm(unique_names, desc="Wikidata queries"):
                    wikidata_results[name] = self.fetch_party_info(name, language=language)
                    time.sleep(0.1)

                # Match the party name with ideology from reference data if available
                if self.ideology_reference_df is not None:
                    ideology_lookup = {r['party']: r['lrgen_category'] for r in self.ideology_reference_df}
                    for name, result in wikidata_results.items():
                        party_name = result.get('party_name_short')
                        party_name = self._normalize_string(party_name) if party_name else None
                        result['lrgen_category'] = ideology_lookup.get(party_name) if party_name else None

                # Apply results to rows
                for row in political_actors:
                    if row['party'] is None:
                        name = row['core_actor_name']
                        if name in wikidata_results:
                            result = wikidata_results[name]
                            if result.get('party_name_short') is not None:
                                row['party'] = result['party_name_short']
                            if result.get('lrgen_category') is not None:
                                row['lrgen_category'] = result['lrgen_category']

                # Add wikidata results to the party reference list for future use
                print("Updating party reference data with Wikidata results...")
                if self.politician_reference_df is not None:
                    existing_names = {ref['name'] for ref in self.politician_reference_df}
                    for name, result in wikidata_results.items():
                        if name not in existing_names:
                            self.politician_reference_df.append({
                                'name': name,
                                'party': result.get('party_name_short'),
                                'lrgen_category': result.get('lrgen_category'),
                            })

                # Save updated politician reference data if path is set
                if self.actor_data_path is not None:
                    politician_ref_path = os.path.splitext(self.actor_data_path)[0] + '_politicians_updated.csv'
                    print(f"Writing updated politician reference data to {politician_ref_path}...")
                    pl.DataFrame(self.politician_reference_df).write_csv(politician_ref_path, separator=';')

        # drop if lrgen_category is missing
        political_actors = [row for row in political_actors if row.get('lrgen_category') is not None]

        return pl.DataFrame(political_actors) if political_actors else pl.DataFrame()

    def calculate_actors_per_partyideology(self, political_actors_df: pl.DataFrame, center_parties: bool) -> pl.DataFrame:
        """
        Calculate number of unique actors per party ideology category for each article.

        Args:
            political_actors_df: DataFrame with enriched political actors (must include id_column and lrgen_category)

        Returns:
            DataFrame with columns: id_column, nr_actors_left, nr_actors_right, nr_actors_center,
            nr_actors_political
        """

        if political_actors_df.is_empty():
            print("Political actors DataFrame is empty. Returning empty DataFrame.")
            return pl.DataFrame()

        # Keep only valid ideology categories
        valid_categories = ['left', 'right']
        if center_parties:
            valid_categories.append('center')

        political_actors_df = political_actors_df.filter(pl.col('lrgen_category').is_in(valid_categories))

        if political_actors_df.is_empty():
            print("No actors with valid ideology categories found.")
            return pl.DataFrame()

        # Count unique actor names per (id, ideology category)
        unique_counts = (
            political_actors_df
            .group_by([self.id_column, 'lrgen_category'])
            .agg(pl.col('actor_name').n_unique().alias('nr_actors'))
        )

        # Pivot to have ideology categories as columns
        ideology_df = unique_counts.pivot(
            index=self.id_column,
            on='lrgen_category',
            values='nr_actors',
        ).fill_null(0)

        # if one of the ideology columns is missing, add it with zeros
        for category in valid_categories:
            if category not in ideology_df.columns:
                ideology_df = ideology_df.with_columns(pl.lit(0).alias(category))

        # Rename columns
        rename_map = {cat: f'nr_actors_{cat}' for cat in valid_categories if cat in ideology_df.columns}
        ideology_df = ideology_df.rename(rename_map)

        # Calculate total number of unique actors
        actor_cols = [c for c in ideology_df.columns if c.startswith('nr_actors_')]
        ideology_df = ideology_df.with_columns(
            pl.sum_horizontal([pl.col(c) for c in actor_cols]).alias('nr_actors_political')
        )

        return ideology_df


    def run_full_enrichment(
        self,
        use_wikidata: bool = True,
        language: str = "en"
    ) -> pl.DataFrame:
        """
        Run the complete enrichment pipeline.

        Args:
            use_wikidata: Whether to query Wikidata for party information
            language: Language code for Wikidata queries

        Returns:
            DataFrame with columns: id_column, nr_actors_a, nr_actors_b, nr_actors_c,
            nr_actors_d, nr_actors_total, nr_actors_left, nr_actors_right,
            nr_actors_center (if center_parties=True), nr_actors_political
        """
        print("\n" + "="*60)
        print("STARTING FULL ACTOR ENRICHMENT PIPELINE")
        print("="*60 + "\n")

        # Step 1: Expand actors to rows
        print("Step 1: Expanding actors to row-level...")
        expanded_df = self.expand_actors_to_rows()
        print(f"Expanded to {len(expanded_df)} actor records\n")

        # Step 2: Calculate actor statistics per function
        print("Step 2: Calculating actor statistics per function...")
        functions_df = self.calculate_actors_per_function(expanded_df)
        print(f"Generated statistics for {len(functions_df)} articles\n")

        # Step 3: Enrich political actors
        print("Step 3: Enriching political actors...")
        enriched_political_df = self.enrich_political_actors(
            expanded_df,
            use_wikidata=use_wikidata,
            language=language
        )
        print(f"Enriched {len(enriched_political_df)} political actor records\n")

        # Step 4: Calculate actors per party ideology
        print("Step 4: Calculating actors per party ideology...")
        ideology_df = self.calculate_actors_per_partyideology(
            enriched_political_df,
            center_parties=self.center_parties
        )
        print(f"Generated ideology statistics for {len(ideology_df)} articles\n")

        print("="*60)
        print("ENRICHMENT PIPELINE COMPLETED")
        print("="*60 + "\n")

        if functions_df.is_empty():
            return ideology_df
        if ideology_df.is_empty():
            return functions_df
        return functions_df.join(ideology_df, on=self.id_column, how='left')


def main(args):
    """Main execution function for command-line usage."""

    # Initialize enricher
    enricher = ActorEnricher(
        actor_data_path=args.actor_data_path,
        id_column=args.id_column,
        language=args.language,
        politicians_data_path=args.politicians_data_path,
    )

    # Run full enrichment pipeline
    actor_stats_df = enricher.run_full_enrichment(
        use_wikidata=args.use_wikidata,
        language=args.language
    )

    # Save outputs
    output_dir = os.path.dirname(args.output_prefix) or '.'
    os.makedirs(output_dir, exist_ok=True)

    # Save actor statistics
    stats_path = f"{args.output_prefix}_actor_stats.csv"
    actor_stats_df.write_csv(stats_path, separator=';')
    print(f"Saved actor statistics to: {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich actors from LLM-annotated DataFrame"
    )

    # Input/output arguments
    parser.add_argument(
        "--actor_data_path",
        type=str,
        required=True,
        help="Path to input CSV file with LLM actor annotations"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="actors_output",
        help="Prefix for output CSV files (will create _expanded.csv, _functions.csv, _political.csv, _ideology.csv)"
    )
    parser.add_argument(
        "--id_column",
        type=str,
        default="uri",
        help="Column name for unique article identifier (default: 'uri')"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language for NER processing (default: 'en'). Use 'nl' for Dutch."
    )
    parser.add_argument(
        "--use_wikidata",
        action="store_true",
        help="Query Wikidata for missing party information"
    )
    parser.add_argument(
        "--politicians_data_path",
        type=str,
        help="Path to CSV with party reference data (columns: name, party, lrgen_category)"
    )

    args = parser.parse_args()
    main(args)