import pandas as pd
import json
import csv
import re
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
            party_reference_path: Optional[str] = None,
            ideology_reference_path: Optional[str] = None):
        
        """
        Initialize ActorEnricher.
        
        Args:
            actor_data_path: Path to CSV file with actor annotations, optional
            actor_df: DataFrame with actor annotations (alternative to path if path not provided)
            id_column: Name of the column containing unique article identifiers
            language: Language for NER processing ('en', 'nl', etc.)
            party_reference_path: Path to CSV with party reference data (columns: person_name, party_short, party_name)
            ideology_reference_path: Path to CSV with party ideology scores (columns: party, lrgen, lrecon, galtan)
        """

        self.actor_data_path = actor_data_path
        self.id_column = id_column
        self.language = language

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
        self.party_reference_df = self._load_party_reference(party_reference_path)
        self.ideology_reference_df = self._load_ideology_reference(ideology_reference_path)


    def _load_actor_data(self) -> pd.DataFrame:
            """Load actor data from CSV file into a DataFrame"""
            if self.actor_data_path:
                return pd.read_csv(
                    self.actor_data_path, 
                    index=False, 
                    sep=';', 
                    quoting=csv.QUOTE_NONNUMERIC)
            else:
                raise ValueError("No actor data path provided and no DataFrame was passed.")
            
    def _load_party_reference(self, path: Optional[str]) -> Optional[pd.DataFrame]:
        """Load party reference data for matching."""
        if path and os.path.exists(path):
            print(f"Loading party reference data from {path}")
            return pd.read_csv(path, sep=';', quoting=csv.QUOTE_NONNUMERIC)
        return None
    
    def _load_ideology_reference(self, path: Optional[str]) -> Optional[pd.DataFrame]:
        """Load party ideology scores."""
        if path and os.path.exists(path):
            print(f"Loading ideology reference data from {path}")
            return pd.read_csv(path, sep=';', quoting=csv.QUOTE_NONNUMERIC)
        return None
    
    def _parse_actors_json(self, actors_json_str):
        """Parse the JSON string and extract actor lists"""
        cleaned = re.sub(r"\s+", " ", actors_json_str)
        cleaned = re.sub(r"^```(?:json)?|```$", "", 
                         cleaned, 
                         flags=re.IGNORECASE | re.MULTILINE
                         ).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if pd.isna(cleaned):
            return [], [], []
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
                f"Available columns: {list(actor_df.columns)}"
            )
        
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
    def calculate_actors_per_function(self, actor_df: pd.DataFrame, id_column: str) -> pd.DataFrame:
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
        
        # Group by article ID and actor function, count unique actor names
        unique_counts = (
            actor_df
            .groupby([self.id_column, 'actor_function'])['actor_name']
            .nunique()
            .reset_index()
        )
        unique_counts = unique_counts.rename(columns={'actor_name': 'nr_actors'})
        
        # Drop if function is not in ['a', 'b', 'c', 'd']
        actor_df = actor_df[actor_df['actor_function'].isin(['a', 'b', 'c', 'd'])]

        # Group by article ID and actor function, then count unique actor names
        unique_counts = (actor_df
                         .groupby([id_column, 'actor_function'])['actor_name']
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

        # Calculate percentage columns
        for col in actor_cols:
            perc_col = col.replace('nr_actors_', 'perc_actors_')
            functions_df[perc_col] = (
                functions_df[col] / functions_df['nr_actors_total']
            ).fillna(0)
        
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

    def _search_wikidata(self, name: str, language: str = "en") -> Optional[str]:
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

    def get_latest_party_from_wikidata(self, name: str, language: str = "nl"
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
            "party_short": latest["short_name"] or None
        }
    
    def fetch_party_info(self, name: str, language: str = "nl") -> pd.Series:
        """
        Wrapper to safely fetch party information from Wikidata.
        
        Args:
            name: Person's name
            language: Language code
            
        Returns:
            Series with party_name and party_short
        """
        try:
            result = self.get_latest_party_from_wikidata(name, language=language)
            if result:
                return pd.Series({
                    "party_name": result["party_name"],
                    "party_short": result["party_short"]
                })
        except Exception as e:
            print(f"Error fetching party info for {name}: {e}")
        
        return pd.Series({"party_name": None, "party_short": None})


    def enrich_political_actors(self, actor_df: Optional[pd.DataFrame] = None,
        use_wikidata: bool = True, wikidata_language: str = "nl") -> pd.DataFrame:
        """        
        Enrichment pipeline for politicians (function 'a', NER person):
        1. Extract core names using NER
        2. Match against party reference data (if available)
        3. Query Wikidata for missing information (if use_wikidata=True)
        4. Merge with ideology scores (if available)
        
        Args:
            actor_df: Optional actor-level DataFrame. If None, expands self.actor_df
            use_wikidata: Whether to query Wikidata for missing party info
            wikidata_language: Language code for Wikidata queries
            
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
        
        # Step 1: Extract core names using NER
        print("Extracting core names using NER...")
        tqdm.pandas(desc="Extracting names")
        political_actors['core_actor_name'] = political_actors['actor_name'].progress_apply(
            self.extract_core_name
        )
        
        # Remove rows where core name extraction failed
        political_actors = political_actors[
            political_actors['core_actor_name'].notna()
        ].copy()
        
        if political_actors.empty:
            print("No valid person names extracted.")
            return pd.DataFrame()
        
        print(f"Extracted {len(political_actors)} valid person names")

        political_actors['core_actor_name'] = political_actors['core_actor_name'].str.title().str.strip()
        
        # Step 2: Match against party reference data
        if self.party_reference_df is not None:
            print("Matching against party reference data...")
            self.party_reference_df['person_name'] = self.party_reference_df['person_name'].str.title().str.strip()
            political_actors = political_actors.merge(
                self.party_reference_df[['person_name', 'party_short', 'party_name']],
                left_on='core_actor_name',
                right_on='person_name',
                how='left'
            )
            political_actors = political_actors.drop(columns=['person_name'], errors='ignore')
            
            matched_count = political_actors['party_name'].notna().sum()
            print(f"Matched {matched_count} actors with reference data")
        else:
            political_actors['party_name'] = None
            political_actors['party_short'] = None
        
        # Step 3: Query Wikidata for missing information
        if use_wikidata:
            missing_party = political_actors['party_name'].isna()
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
                    wikidata_results[name] = self.fetch_party_info(
                        name, 
                        language=wikidata_language
                    )
                    # Small delay to avoid rate limiting
                    import time
                    time.sleep(0.1)
                
                # Apply results to DataFrame
                for idx in political_actors[missing_party].index:
                    name = political_actors.at[idx, 'core_actor_name']
                    if name in wikidata_results:
                        result = wikidata_results[name]
                        if pd.notna(result['party_name']):
                            political_actors.at[idx, 'party_name'] = result['party_name']
                            political_actors.at[idx, 'party_short'] = result['party_short']

                # Add wikidata results to the party reference df for future use
                print("Updating party reference data with Wikidata results...")
                wikidata_df = pd.DataFrame.from_dict(wikidata_results, orient='index').reset_index()
                wikidata_df = wikidata_df.rename(columns={'index': 'person_name'})
                self.party_reference_df = pd.concat([self.party_reference_df, wikidata_df], 
                                                    ignore_index=True)
                
                # write updated party reference back to file if path provided
                if self.party_reference_df is not None and self.actor_data_path is not None:
                    party_ref_path = os.path.splitext(self.actor_data_path)[0] + '_party_reference.csv'
                    print(f"Writing updated party reference data to {party_ref_path}...")
                    self.party_reference_df.to_csv(
                        party_ref_path, 
                        sep=';', 
                        quoting=csv.QUOTE_NONNUMERIC, 
                        index=False
                    )
        
        # Step 4: Merge with ideology scores
        if self.ideology_reference_df is not None:
            print("Merging with ideology scores...")
            political_actors = political_actors.merge(
                self.ideology_reference_df,
                left_on='party_short',
                right_on='party',
                how='left'
            )
            political_actors = political_actors.drop(columns=['party'], errors='ignore')
            
            ideology_matched = political_actors['lrgen'].notna().sum()
            print(f"Matched {ideology_matched} actors with ideology scores")
        
        return political_actors


    def run_full_enrichment(
        self,
        use_wikidata: bool = True,
        wikidata_language: str = "nl"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the complete enrichment pipeline.
        
        Args:
            use_wikidata: Whether to query Wikidata for party information
            wikidata_language: Language code for Wikidata queries
            
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
            wikidata_language=wikidata_language
        )
        print(f"Enriched {len(enriched_political_df)} political actor records\n")
        
        print("="*60)
        print("ENRICHMENT PIPELINE COMPLETED")
        print("="*60 + "\n")
        
        return expanded_df, functions_df, enriched_political_df


def main(args):
    """Main execution function for command-line usage."""
    
    # Initialize enricher
    enricher = ActorEnricher(
        actor_data_path=args.actor_data_path,
        id_column=args.id_column,
        language=args.language,
        party_reference_path=args.party_reference_path,
        ideology_reference_path=args.ideology_reference_path
    )

    # Run full enrichment pipeline
    expanded_df, functions_df, enriched_political_df = enricher.run_full_enrichment(
        use_wikidata=args.use_wikidata,
        wikidata_language=args.wikidata_language
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
    political_path = f"{args.output_prefix}_political.csv"
    enriched_political_df.to_csv(
        political_path, 
        index=False, 
        sep=';', 
        quoting=csv.QUOTE_NONNUMERIC
    )
    print(f"Saved enriched political actors to: {political_path}")


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
        required=True,
        help="Prefix for output CSV files (will create _expanded.csv, _functions.csv, _political.csv)"
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
    parser.add_argument(
        "--wikidata_language",
        type=str,
        default="nl",
        help="Language code for Wikidata queries (default: 'nl')"
    )
    
    # Reference data arguments
    parser.add_argument(
        "--party_reference_path",
        type=str,
        help="Path to CSV with party reference data (columns: person_name, party_short, party_name)"
    )
    parser.add_argument(
        "--ideology_reference_path",
        type=str,
        help="Path to CSV with party ideology scores (columns: party, lrgen, lrecon, galtan)"
    )
    
    args = parser.parse_args()
    main(args)