import pandas as pd
import json
import csv
import re
import unicodedata
import os
import argparse
from typing import List, Dict, Optional, Tuple
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
            actor_df: Optional[pd.DataFrame] = None, 
            id_column: str = 'news_id', 
            language: str = 'en',
            center_parties: bool = True,
            save_politicians_df: bool = False,
            political_data_path: Optional[str] = None):
        
        """
        Initialize ActorEnricher.
        
        Args:
            actor_data_path: Path to CSV file with actor annotations, optional
            actor_df: DataFrame with actor annotations (alternative to path if path not provided)
            id_column: Name of the column containing unique article identifiers
            language: Language for NER processing ('en', 'nl', etc.)
            center_parties: Whether the country has center parties in addition to left and right
            save_politicians_df: Whether to save enriched political actors to _political.csv file
            political_data_path: Path to CSV with party reference data (columns: name, party, lrgen_category)
        """

        self.actor_data_path = actor_data_path
        self.id_column = id_column
        self.language = language
        self.center_parties = center_parties
        self.save_politicians_df = save_politicians_df

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

        # Load reference data
        self.political_df = self._load_political_data(political_data_path)
        self.politician_reference_df = self.political_df[['name', 'party', 'lrgen_category']].drop_duplicates() if self.political_df is not None else None
        self.ideology_reference_df = self.political_df[['party', 'lrgen_category']].drop_duplicates() if self.political_df is not None else None

        # add two parties to ideology reference data if not already present: GROENLINKS-PVDA left, and CU center
        if self.language == 'nl' and self.ideology_reference_df is not None:
            additional_parties = pd.DataFrame([
                    {'party': 'GROENLINKS-PVDA', 'lrgen_category': 'left'},
                    {'party': 'GROENLINKS', 'lrgen_category': 'left'},
                    {'party': 'PVDA', 'lrgen_category': 'left'},
                    {'party': 'CU', 'lrgen_category': 'center'}
                ])
            self.ideology_reference_df = pd.concat([self.ideology_reference_df, additional_parties], ignore_index=True).drop_duplicates(subset=['party'], keep='first')

    def _load_actor_data(self) -> pd.DataFrame:
            """Load actor data from CSV file into a DataFrame"""
            if self.actor_data_path:
                return pd.read_csv(
                    self.actor_data_path, 
                    index_col=False, 
                    sep=';', 
                    quoting=csv.QUOTE_NONNUMERIC)
            else:
                raise ValueError("No actor data path provided and no DataFrame was passed.")
            
    def _load_political_data(self, path: Optional[str]) -> Optional[pd.DataFrame]:
        """Load party reference data for matching."""
        if path and os.path.exists(path):
            print(f"Loading party reference data from {path}")
            df = pd.read_csv(path, sep=';', quoting=csv.QUOTE_NONNUMERIC)
            df = df.drop_duplicates(subset=['name'], keep='first')
            return df
        return None

    def _parse_actors_json(self, actors_json_str):
        if pd.isna(actors_json_str):
            return [], [], []
        
        """Parse the JSON string and extract actor lists"""
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
    
    def expand_actors_to_rows(self, actor_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
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
            
        if actor_df.empty:
            return pd.DataFrame()
        
        # Validate that the ID column exists
        if self.id_column not in actor_df.columns:
            raise ValueError(
                f"ID column '{self.id_column}' not found in DataFrame. "
                f"Available columns: {list(actor_df.columns)}")
        
        def generate_actor_rows():
            """Generator that yields actor rows one at a time."""
            for idx, row in actor_df.iterrows():
                # Parse actors for this row
                names, functions, parties = self._parse_actors_json(row['news_actors'])  
                raw_output = row.get('news_actors_raw', '')
                
                # If no actors found, create one row with empty values
                if len(names) == 0:
                    yield {
                        self.id_column: row[self.id_column],
                        'actor_name': '',
                        'actor_function': '',
                        'actor_pp': '',
                        'news_actors_raw': raw_output
                        }
                else:
                    # Each actor as a separate row
                    for i in range(len(names)):
                        yield {
                            self.id_column: row[self.id_column],
                            'actor_name': names[i] if i < len(names) else '',
                            'actor_function': functions[i] if i < len(functions) else '',
                            'actor_pp': parties[i] if i < len(parties) else '',
                            'news_actors_raw': raw_output
                        }
        
        rows = list(generate_actor_rows())  
        return pd.DataFrame(rows)
    
    # calculate nr of unique actors per function per article
    def calculate_actors_per_function(self, actor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate number of unique actors per function for each article.
        
        Args:
            actor_df: Optional actor-level DataFrame. If None, expands self.actor_df
            
        Returns:
            DataFrame with columns: id_column, nr_actors_a, nr_actors_b, nr_actors_c, 
            nr_actors_d, nr_actors_total, perc_actors_a, perc_actors_b, perc_actors_c, 
            perc_actors_d
        """
        if actor_df is None:
            actor_df = self.expand_actors_to_rows()
        
        if actor_df.empty:
            print("Actor DataFrame is empty. Returning empty DataFrame.")
            return pd.DataFrame()
        
        # Keep only valid functions
        valid_functions = ['a', 'b', 'c', 'd']
        actor_df = actor_df[actor_df['actor_function'].isin(valid_functions)]
        
        if actor_df.empty:
            print("No actors with valid functions (a, b, c, d) found.")
            return pd.DataFrame()
        
        # Group by article ID and actor function, then count unique actor names
        unique_counts = (actor_df
                         .groupby([self.id_column, 'actor_function'])['actor_name']
                         .nunique()
                         .reset_index())
        unique_counts = unique_counts.rename(columns={'actor_name': 'nr_actors'})
        
        # Pivot to have functions as columns
        functions_df = (
            unique_counts
            .pivot(
                index=self.id_column, 
                columns='actor_function', 
                values='nr_actors'
            )
            .fillna(0)
            .reset_index()
        )

        # if one of the function columns is missing, add it with zeros
        for func in valid_functions:
            if func not in functions_df.columns:
                functions_df[func] = 0
        
        # Rename columns
        functions_df = functions_df.rename(columns={
            'a': 'nr_actors_a', 
            'b': 'nr_actors_b', 
            'c': 'nr_actors_c', 
            'd': 'nr_actors_d'
        })

        # Calculate total number of unique actors
        actor_cols = [col for col in functions_df.columns if col.startswith('nr_actors_')]
        functions_df['nr_actors_total'] = functions_df[actor_cols].sum(axis=1)
        
        return functions_df
        
    def _clean_actor_name(self, name: str) -> str:
        """Remove text in parentheses and extra whitespace."""
        if pd.isna(name):
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
        if pd.isna(full_name) or not str(full_name).strip():
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
            Dictionary with 'party' and 'short_name' keys, or None if not found
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
        
        df = pd.DataFrame([{
            "party": r["partyLabel"]["value"],
            "short_name": r.get("shortName", {}).get("value"),
            "start": r.get("start", {}).get("value"),
            "end": r.get("end", {}).get("value"),
        } for r in results["results"]["bindings"]])
        
        if df.empty:
            return None
        
        # Order by start descending, then end descending
        df['start'] = pd.to_datetime(df['start'], errors='coerce')
        df['end'] = pd.to_datetime(df['end'], errors='coerce')
        df = df.sort_values(
            by=['start', 'end'], 
            ascending=[False, False]
        ).reset_index(drop=True)
        
        latest = df.iloc[0]
        return {
            "party_name": latest["party"],
            "party_name_short": latest["short_name"] or None
        }
    
    def fetch_party_info(self, name: str, language: str = "en") -> pd.Series:
        """
        Wrapper to safely fetch party information from Wikidata.
        
        Args:
            name: Person's name
            language: Language code
            
        Returns:
            Series with party_name and party_name_short
        """
        try:
            result = self.get_latest_party_from_wikidata(name, language=language)
            if result:
                return pd.Series({
                    "party_name": result["party_name"],
                    "party_name_short": result["party_name_short"]
                })
        except Exception as e:
            print(f"Error fetching party info for {name}: {e}")
        
        return pd.Series({"party_name": None, "party_name_short": None})

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
                                actor_df: Optional[pd.DataFrame] = None,
                                use_wikidata: bool = True, 
                                language: str = "en") -> pd.DataFrame:
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
        
        if actor_df.empty:
            print("Actor DataFrame is empty. Returning empty DataFrame.")
            return pd.DataFrame()
        
        # Filter for political actors only (function 'a')
        political_actors = actor_df[actor_df['actor_function'] == 'a'].copy()
        
        if political_actors.empty:
            print("No political actors (function 'a') found.")
            return pd.DataFrame()
        
        print(f"Processing {len(political_actors)} political actor records...")

        political_actors['actor_name_upper'] = political_actors['actor_name'].apply(self._normalize_string)
        if self.politician_reference_df is not None:
            self.politician_reference_df['name'] = self.politician_reference_df['name'].apply(self._normalize_string)
        
        # Step 1: Check for party name mentions in actor_name (before NER)
        print("Step 1: Checking for party name mentions in actor names...")
        political_actors['party'] = None
        political_actors['lrgen_category'] = None
        political_actors['matched_name'] = None
        party_mention_count = 0
        new_rows = []

        if self.ideology_reference_df is not None:
            for idx in political_actors.index:
                actor_name = str(political_actors.at[idx, 'actor_name_upper']) 
                actor_name = re.sub(r"[-–—/]", " ", actor_name)
                actor_name = re.sub(r"\s+", " ", actor_name).strip()

                matched_parties = []
                
                for _, ref_row in self.ideology_reference_df.iterrows():
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
                    # Update original row with first match
                    political_actors.at[idx, 'party'] = matched_parties[0]['party']
                    political_actors.at[idx, 'lrgen_category'] = matched_parties[0]['lrgen_category']
                    political_actors.at[idx, 'matched_name'] = matched_parties[0]['matched_name']
                    for match in matched_parties[1:]:
                        row_copy = political_actors.loc[idx].copy()
                        row_copy['party'] = match['party']
                        row_copy['lrgen_category'] = match['lrgen_category']
                        row_copy['matched_name'] = match['matched_name']
                        new_rows.append(row_copy)

            # Append new rows for additional matches
            if new_rows:
                new_rows_df = pd.DataFrame(new_rows)
                political_actors = pd.concat([political_actors, new_rows_df], ignore_index=True)            
            
        print(f". → Matched {party_mention_count} actors with party name mentions")

        # Initialize column for all rows
        political_actors['core_actor_name'] = None

        unmatched_mask = political_actors['party'].isna()
        unmatched_count = unmatched_mask.sum()
        print(f"Step 2: Extracting core names using NER for {unmatched_count} actors with no party mentions...")

        if unmatched_count > 0:
            tqdm.pandas(desc="Extracting names")
            ner_results = political_actors.loc[unmatched_mask, 'actor_name'].progress_apply(
                self.extract_core_name
            )
            political_actors.loc[unmatched_mask, 'core_actor_name'] = ner_results

        political_actors['core_actor_name_upper'] = political_actors['core_actor_name'].apply(
            lambda x: self._normalize_string(x) if pd.notna(x) else None
        )

        if self.politician_reference_df is not None:
            print("Step 2.1: Exact matching on core_actor_name...")
            matched_mask = political_actors['party'].notna()
            step1_matched_rows = matched_mask.sum()
            matched_df = political_actors[matched_mask].copy()
            unmatched_df = political_actors[~matched_mask].copy()
            
            unmatched_df = unmatched_df.merge(
                self.politician_reference_df,
                left_on='core_actor_name_upper',
                right_on='name',
                how='left',
                suffixes=('', '_ref')
            )
            unmatched_df['party'] = unmatched_df['party_ref']
            unmatched_df['lrgen_category'] = unmatched_df['lrgen_category_ref']
            unmatched_df['matched_name'] = unmatched_df['name']
            unmatched_df.drop(columns=['party_ref', 'lrgen_category_ref', 'name'], inplace=True)

            political_actors = pd.concat([matched_df, unmatched_df], ignore_index=True)
                    
            exact_count = political_actors['party'].notna().sum() - step1_matched_rows  
            print(f"  → Matched {exact_count} actors with exact core_actor_name match")
            print(f"These names are matched with reference data: {political_actors.loc[political_actors['party'].notna(), ['core_actor_name', 'matched_name']].drop_duplicates().to_dict(orient='records')}")
            
        # Step 2.2: Token match on core_actor_name (for unmatched rows)
        if self.politician_reference_df is not None:
            print("Step 2.2: Token matching on core_actor_name...")
            unmatched_mask = political_actors['party'].isna()
            token_match_count = 0
            
            for idx in political_actors[unmatched_mask].index:
                core_name = political_actors.at[idx, 'core_actor_name_upper']
                for _, ref_row in self.politician_reference_df.iterrows():
                    if self._surname_match(core_name, ref_row['name']):
                        political_actors.at[idx, 'party'] = ref_row['party']
                        political_actors.at[idx, 'lrgen_category'] = ref_row['lrgen_category']
                        political_actors.at[idx, 'matched_name'] = ref_row['name']
                        token_match_count += 1
                        break

            print(f"→ Matched {token_match_count} actors with token match on core_actor_name")
            print(f"These names are matched with reference data: {political_actors.loc[political_actors['party'].notna(), ['core_actor_name', 'matched_name']].drop_duplicates().to_dict(orient='records')}")
                
            # Step 2.3: Exact + token match on actor_name (for still unmatched rows)
            print("Step 2.3: Matching on original actor_name...")
            unmatched_mask = political_actors['party'].isna()
            actor_name_count = 0
            
            for idx in political_actors[unmatched_mask].index:
                actor_name = political_actors.at[idx, 'actor_name_upper']
                # Try exact match first
                exact_ref = self.politician_reference_df[self.politician_reference_df['name'] == actor_name]
                if not exact_ref.empty:
                    political_actors.at[idx, 'party'] = exact_ref.iloc[0]['party']
                    political_actors.at[idx, 'lrgen_category'] = exact_ref.iloc[0]['lrgen_category']
                    political_actors.at[idx, 'matched_name'] = exact_ref.iloc[0]['name']
                    actor_name_count += 1
                    continue
                
                # Try token match as well
                for _, ref_row in self.politician_reference_df.iterrows():
                    if self._surname_match(actor_name, ref_row['name']):
                        political_actors.at[idx, 'party'] = ref_row['party']
                        political_actors.at[idx, 'lrgen_category'] = ref_row['lrgen_category']
                        political_actors.at[idx, 'matched_name'] = ref_row['name']
                        actor_name_count += 1
                        break
            
            print(f"→ Matched {actor_name_count} actors with actor_name matching")
                        
            total_matched = political_actors['party'].notna().sum()
            print(f"\nTotal matched: {total_matched} actors with reference data")

        # Step 3: Query Wikidata for missing information
        if use_wikidata:
            missing_party = political_actors['party'].isna()
            missing_count = missing_party.sum()
            
            if missing_count > 0:
                print(f"Querying Wikidata for {missing_count} actors with missing party info...")
                
                # Get unique names to avoid duplicate queries
                unique_missing = political_actors.loc[
                    missing_party, 'core_actor_name'
                ].unique()
                
                print(f"Querying {len(unique_missing)} unique names...")
                
                # Query Wikidata with progress bar
                wikidata_results = {}                    
                for name in tqdm(unique_missing, desc="Wikidata queries"):
                    if name is None:
                        continue
                    wikidata_results[name] = self.fetch_party_info(name, language=language)
                    # Small delay to avoid rate limiting
                    import time
                    time.sleep(0.1)

                # Match the party name with ideology from reference data if available
                if self.ideology_reference_df is not None:
                    for name, result in wikidata_results.items():
                        party_name = result.get('party_name_short')
                        # normalize party name for matching
                        party_name = self._normalize_string(party_name) if party_name else None
                        if party_name:
                            ideology_row = self.ideology_reference_df[
                                self.ideology_reference_df['party'] == party_name
                            ]
                            if not ideology_row.empty:
                                result['lrgen_category'] = ideology_row.iloc[0]['lrgen_category']
                            else:
                                result['lrgen_category'] = None
                        else:
                            result['lrgen_category'] = None
                
                # Apply results to DataFrame
                for idx in political_actors[missing_party].index:
                    name = political_actors.at[idx, 'core_actor_name']
                    if name in wikidata_results:
                        result = wikidata_results[name]
                        if pd.notna(result['party_name_short']):
                            political_actors.at[idx, 'party'] = result['party_name_short']
                        if pd.notna(result.get('lrgen_category')):
                            political_actors.at[idx, 'lrgen_category'] = result['lrgen_category']

                # Add wikidata results to the party reference df for future use
                print("Updating party reference data with Wikidata results...")
                wikidata_df = pd.DataFrame.from_dict(wikidata_results, orient='index').reset_index()
                wikidata_df = wikidata_df.rename(columns={'index': 'name', 
                                                          'party': 'party_name_short'})
                
                self.politician_reference_df = pd.concat([self.politician_reference_df, wikidata_df], 
                                                    ignore_index=True)
                
                # Save updated politician reference data if path is set
                if self.actor_data_path is not None:
                    politician_ref_path = os.path.splitext(self.actor_data_path)[0] + '_politicians_updated.csv'
                    print(f"Writing updated politician reference data to {politician_ref_path}...")
                    self.politician_reference_df.to_csv(
                        politician_ref_path, 
                        sep=';', 
                        quoting=csv.QUOTE_NONNUMERIC, 
                        index=False
                    )
                

        # drop if lrgen_category is missing
        political_actors = political_actors[political_actors['lrgen_category'].notna()].copy()
        
        return political_actors
    
    def calculate_actors_per_partyideology(self, political_actors_df: pd.DataFrame, center_parties: bool) -> pd.DataFrame:
        """
        Calculate number of unique actors per party ideology category for each article.
        
        Args:
            political_actors_df: DataFrame with enriched political actors (must include id_column and lrgen_category)

        Returns:
            DataFrame with columns: id_column, nr_actors_left, nr_actors_right, nr_actors_center, 
            nr_actors_total, perc_actors_left, perc_actors_right, perc_actors_center
        """

        if political_actors_df.empty:
            print("Political actors DataFrame is empty. Returning empty DataFrame.")
            return pd.DataFrame()
        
        # Keep only valid ideology categories
        valid_categories = ['left', 'right']
        if center_parties:
            valid_categories.append('center')
        
        political_actors_df = political_actors_df[political_actors_df['lrgen_category'].isin(valid_categories)]
        
        if political_actors_df.empty:
            print("No actors with valid ideology categories found.")
            return pd.DataFrame()
        
        # Group by article ID and ideology category, then count unique actor names
        unique_counts = (political_actors_df
                         .groupby([self.id_column, 'lrgen_category'])['actor_name']
                         .nunique()
                         .reset_index())
        unique_counts = unique_counts.rename(columns={'actor_name': 'nr_actors'})
        
        # Pivot to have ideology categories as columns
        ideology_df = (
            unique_counts
            .pivot(
                index=self.id_column, 
                columns='lrgen_category', 
                values='nr_actors'
            )
            .fillna(0)
            .reset_index()
        )

        # if one of the ideology columns is missing, add it with zeros
        for category in valid_categories:
            col_name = f"nr_actors_{category}"
            if category not in ideology_df.columns:
                ideology_df[col_name] = 0
            else:
                ideology_df.rename(columns={category: col_name}, inplace=True)
        
        # Calculate total number of unique actors
        actor_cols = [col for col in ideology_df.columns if col.startswith('nr_actors_')]
        ideology_df['nr_actors_total'] = ideology_df[actor_cols].sum(axis=1)
        
        return ideology_df


    def run_full_enrichment(
        self,
        use_wikidata: bool = True,
        language: str = "en"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the complete enrichment pipeline.
        
        Args:
            use_wikidata: Whether to query Wikidata for party information
            language: Language code for Wikidata queries
            
        Returns:
            Tuple of (expanded_df, functions_df, enriched_political_df)
            - expanded_df: All actors at row level
            - functions_df: Actor count statistics per function per article
            - enriched_political_df: Political actors with party and ideology info
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
            center_parties=True
        )
        print(f"Generated ideology statistics for {len(ideology_df)} articles\n")
        
        print("="*60)
        print("ENRICHMENT PIPELINE COMPLETED")
        print("="*60 + "\n")
        
        return expanded_df, functions_df, enriched_political_df, ideology_df


def main(args):
    """Main execution function for command-line usage."""
    
    # Initialize enricher
    enricher = ActorEnricher(
        actor_data_path=args.actor_data_path,
        id_column=args.id_column,
        language=args.language,
        political_data_path=args.political_data_path,
        save_politicians_df=args.save_politicians_df
    )

    # Run full enrichment pipeline
    expanded_df, functions_df, enriched_political_df, ideology_df = enricher.run_full_enrichment(
        use_wikidata=args.use_wikidata,
        language=args.language
    )

    # Save outputs
    output_dir = os.path.dirname(args.output_prefix) or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save expanded actors
    expanded_path = f"{args.output_prefix}_expanded.csv"
    expanded_df.to_csv(
        expanded_path, 
        index=False, 
        sep=';', 
        quoting=csv.QUOTE_NONNUMERIC
    )
    print(f"Saved expanded actors to: {expanded_path}")
    
    # Save function statistics
    functions_path = f"{args.output_prefix}_functions.csv"
    functions_df.to_csv(
        functions_path, 
        index=False, 
        sep=';', 
        quoting=csv.QUOTE_NONNUMERIC
    )
    print(f"Saved function statistics to: {functions_path}")
    
    # Save enriched political actors
    if enricher.save_politicians_df: 
        political_path = f"{args.output_prefix}_political.csv"
        enriched_political_df.to_csv(
            political_path, 
            index=False, 
            sep=';', 
            quoting=csv.QUOTE_NONNUMERIC
        )
    print(f"Saved enriched political actors to: {political_path}")
    
    # Save ideology statistics
    ideology_path = f"{args.output_prefix}_ideology.csv"
    ideology_df.to_csv(
        ideology_path, 
        index=False, 
        sep=';', 
        quoting=csv.QUOTE_NONNUMERIC
    )
    print(f"Saved ideology statistics to: {ideology_path}")


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
        default="news_id",
        help="Column name for unique article identifier (default: 'news_id')"
    )
    
    # Language and processing arguments
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
    # Reference data arguments
    parser.add_argument(
        "--political_data_path",
        type=str,
        help="Path to CSV with party reference data (columns: name, party, lrgen_category)"
    )

    parser.add_argument(
        "--save_politicians_df",
        action="store_true",
        help="Save updated politician reference data to file after Wikidata queries"
    )

    args = parser.parse_args()
    main(args)