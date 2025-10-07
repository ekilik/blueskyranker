import pandas as pd
import json
import csv
import re
import os
import argparse
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import ollama

###TODO: currently writes the annotated file, maybe not needed? 
###TODO: hardcoded column names, wait time between model calls, minimum text length

class ActorAnnotator():
    def __init__(self, model_name: str = "gpt-oss:20b", seed: int = 0, gpu: int = 0):
        self.model_name = model_name
        self.seed = seed
        self.gpu = gpu
        self.system_prompt = None
        self.main_prompt = None
        self._model_checked = False
        self._load_prompts()

    def _load_prompts(self):
        """Load system and main prompts from file"""
        prompt_file_path = os.path.join(os.path.dirname(__file__), 'actor_extraction_prompt.txt')
        if not os.path.exists(prompt_file_path):
            print(f"Prompt file not found: {prompt_file_path}")
            return
        
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                prompt_content = f.read()

            if "SYSTEM_PROMPT:" not in prompt_content or "MAIN_PROMPT:" not in prompt_content:
                print("Prompt file is missing required sections.")
                return
            
            parts = prompt_content.split("MAIN_PROMPT:")
            self.system_prompt = parts[0].replace("SYSTEM_PROMPT:", "").strip()
            self.main_prompt = parts[1].strip()
        except Exception as e:
            print(f"Error loading prompts: {e}")

    def _ensure_model_available(self) -> bool:
        """Check if model is available, pull only if needed"""
        if self._model_checked:
            return True
        
        try:
            # Check if model exists first
            print(f"Checking availability of model: {self.model_name}")
            models_response = ollama.list()
            
            # Extract model names from the ListResponse object
            model_names = []
            if hasattr(models_response, 'models'):
                for model in models_response.models:
                    if hasattr(model, 'model'):
                        model_names.append(model.model)
            
            if self.model_name not in model_names:
                print(f"Model {self.model_name} not found. Pulling model...")
                print("This may take several minutes for large models.")
                ollama.pull(self.model_name)
                print(f"Successfully pulled model {self.model_name}")
            else:
                print(f"Model {self.model_name} already available")
            
            self._model_checked = True
            return True
            
        except ImportError:
            print("Error: 'ollama' package not found. Install with: pip install ollama")
            return False
        except Exception as e:
            print(f"Failed to ensure model availability: {type(e).__name__}: {e}")
            return False

    def build_prompt(self, system_prompt: str, main_prompt: str) -> List[Dict[str, str]]:
        """Build message format for Ollama"""
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": main_prompt
            }
        ]
        return messages

    def extract_actors_from_content(self, news_content: str, ollama_model: str = None) -> Optional[str]:
        """Extract news actors from news content using Ollama."""
        if ollama_model is None:
            ollama_model = self.model_name
            
        if not news_content or len(news_content.strip()) < 50:
            return None    

        if not self.system_prompt or not self.main_prompt:
            print("Prompts not loaded properly")
            return None
        
        if not self._ensure_model_available():
            return None

        try:
            prompt = self.build_prompt(
                system_prompt=self.system_prompt, 
                main_prompt=self.main_prompt + "\n" + news_content
            )

            response = ollama.chat(
                model=ollama_model, 
                messages=prompt, 
                options={'temperature': 0.0, 'seed': self.seed}
            )
            response_content = response['message']['content'].strip()
            cleaned_response = self.clean_and_parse_actors(response_content)
            
            return cleaned_response
        
        except Exception as e:
            print(f"Ollama actor extraction failed: {e}")
            return None

    def clean_and_parse_actors(self, llm_response: str) -> Optional[str]:
        """Clean LLM response and extract structured actor data."""
        try:
            actors_pattern = r'"?actors"?\s*:\s*\[(.*?)\]'
            actors_match = re.search(actors_pattern, llm_response, re.DOTALL | re.IGNORECASE)
            
            if not actors_match:
                print("No actors array pattern found in LLM response")
                return None
            
            actors_content = actors_match.group(1).strip()
            if not actors_content:
                print("Empty actors array found")
                return None
            
            try:
                actors_json = f"[{actors_content}]"
                actors_list = json.loads(actors_json)
            except json.JSONDecodeError:
                print("Direct JSON parsing failed, trying pattern extraction")
                actors_list = self.extract_actor_json_from_output(actors_content)
            
            if not actors_list or not isinstance(actors_list, list):
                return None
            
            cleaned_actors = []
            for actor in actors_list:
                if isinstance(actor, dict) and 'actor_name' in actor:
                    cleaned_actor = {
                        'actor_name': str(actor.get('actor_name', '')).strip(),
                        'actor_function': str(actor.get('actor_function', '')).strip(),
                        'actor_pp': str(actor.get('actor_pp', '')).strip()
                    }
                    if cleaned_actor['actor_name']:
                        cleaned_actors.append(cleaned_actor)
            
            return json.dumps({'actors': cleaned_actors}) if cleaned_actors else None
            
        except Exception as e:
            print(f"Failed to clean actor response: {e}")
            return None

    def extract_actor_json_from_output(self, actors_text: str) -> List[Dict[str, str]]:
        """Extract actor dictionaries from text using pattern matching."""
        actors = []
        
        actor_pattern = r'\{[^}]*"?actor_name"?\s*:\s*"([^"]*)"[^}]*"?actor_function"?\s*:\s*"([^"]*)"[^}]*"?actor_pp"?\s*:\s*"([^"]*)"\s*[^}]*\}'
        
        matches = re.findall(actor_pattern, actors_text, re.IGNORECASE)
        
        for match in matches:
            actor_name, actor_function, actor_pp = match
            if actor_name.strip():
                actors.append({
                    'actor_name': actor_name.strip(),
                    'actor_function': actor_function.strip(),
                    'actor_pp': actor_pp.strip()
                })
        
        if not actors:
            actors = self.extract_actor_info_from_text(actors_text)
        
        return actors

    def extract_actor_info_from_text(self, actors_text: str) -> List[Dict[str, str]]:
        """Extract actors from text that might be in a simpler format."""
        actors = []
        lines = actors_text.split('\n')
        
        current_actor = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            name_match = re.search(r'"?actor_name"?\s*:\s*"?([^",\n]+)"?', line, re.IGNORECASE)
            function_match = re.search(r'"?actor_function"?\s*:\s*"?([^",\n]+)"?', line, re.IGNORECASE)
            pp_match = re.search(r'"?actor_pp"?\s*:\s*"?([^",\n]*)"?', line, re.IGNORECASE)
            
            if name_match:
                if current_actor.get('actor_name'):
                    actors.append(current_actor)
                current_actor = {
                    'actor_name': name_match.group(1).strip().strip('"'),
                    'actor_function': '',
                    'actor_pp': ''
                }
            
            if function_match and current_actor:
                current_actor['actor_function'] = function_match.group(1).strip().strip('"')
            
            if pp_match and current_actor:
                current_actor['actor_pp'] = pp_match.group(1).strip().strip('"')
        
        if current_actor.get('actor_name'):
            actors.append(current_actor)
        
        return actors
    
    # Update process_articles_batch to accept text_column parameter
    def process_articles_batch(self, df_batch: pd.DataFrame, text_column: str = 'text') -> List[Optional[str]]:
        """Process a batch of articles and return actor extraction results"""
        results = []
        for idx, row in df_batch.iterrows():
            uri = row.get('uri', None)
            text = row.get(text_column, '')
            print(f"Processing post {uri}...")
            actors_json = self.extract_actors_from_content(text)
            results.append(actors_json)
            time.sleep(0.5)  
        return results
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'article_fulltext', 
                    id_column: str = 'uri') -> pd.DataFrame:
        """
        Process a DataFrame and return it with actors column added.
        For in-memory processing without file output.
        """
        df_result = df.copy()
        
        print(f"Processing {len(df_result)} articles...")
        
        # Process each article
        actors_results = []
        for idx, row in tqdm(df_result.iterrows(), total=len(df_result), desc="Extracting actors"):
            uri = row.get(id_column, idx)
            text = row.get(text_column, '')
            
            print(f"Processing post {uri}...")
            actors_json = self.extract_actors_from_content(text)
            actors_results.append(actors_json)
            
            # Brief pause to avoid overwhelming the model
            time.sleep(0.5)
        
        df_result['news_actors'] = actors_results
        return df_result

    @staticmethod
    def parse_actors_json(actors_json_str):
        """Parse the JSON string and extract actor lists"""
        if pd.isna(actors_json_str):
            return [], [], []
        try:
            data = json.loads(actors_json_str)
            actors = data.get('actors', [])
            
            names = [actor.get('actor_name', '') for actor in actors]
            functions = [actor.get('actor_function', '') for actor in actors] 
            parties = [actor.get('actor_pp', '') for actor in actors]
            
            return names, functions, parties
        except (json.JSONDecodeError, AttributeError):
            return [], [], []
        
    @staticmethod
    def expand_actors_to_rows(df):
        """
        Transform DataFrame from article-level to actor-level.
        Each actor becomes a separate row with article metadata.
        """

        df_copy = df.copy()
        parsed_data = df_copy['news_actors'].apply(ActorAnnotator.parse_actors_json)
        
        df_copy['actor_name'] = [x[0] for x in parsed_data]
        df_copy['actor_function'] = [x[1] for x in parsed_data] 
        df_copy['actor_pp'] = [x[2] for x in parsed_data]
        
        expanded_rows = []
        
        for idx, row in df_copy.iterrows():
            # Skip rows with no actors
            if pd.isna(row['news_actors']) or len(row['actor_name']) == 0:
                continue
                
            # Get the number of actors for this article
            num_actors = len(row['actor_name'])
            
            # Create a row for each actor
            for i in range(num_actors):
                actor_row = {
                    'uri': row['uri'],
                    'text': row['text'],
                    'news_title': row['news_title'],
                    'news_description': row['news_description'],
                    'news_uri': row['news_uri'],
                    'news_content': row['news_content'],
                    # Add individual actor data
                    'actor_name': row['actor_name'][i] if i < len(row['actor_name']) else '',
                    'actor_function': row['actor_function'][i] if i < len(row['actor_function']) else '',
                    'actor_pp': row['actor_pp'][i] if i < len(row['actor_pp']) else '',
                    # Add actor index for reference
                    'actor_index': i + 1,
                    'total_actors_in_article': num_actors
                }
                expanded_rows.append(actor_row)
        
        return pd.DataFrame(expanded_rows)
    

    def chunk_dataframe(self, df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
        """Split dataframe into chunks for batch processing"""
        return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    

    def process_all_batches(self, df: pd.DataFrame, batch_size: int, 
                        output_file_path: str, text_column: str = 'full_text') -> None:
        """Process all batches, expand actors, and save reorganized results to CSV file"""
        # Prepare unique articles
        unique_articles = df[['uri', text_column]].drop_duplicates(subset=['uri']).reset_index(drop=True)
        
        # Keep other columns for merging back
        article_metadata = df.drop_duplicates(subset=['uri']).set_index('uri')
        
        batches = self.chunk_dataframe(unique_articles, batch_size)

        # Free memory from original dataframe
        del df
        
        # Remove existing output file if it exists to start fresh
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        
        # Track if we've written headers yet
        header_written = False
        
        for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
            print(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} articles)")
            
            # Step 1: Extract actors
            batch_results = self.process_articles_batch(batch, text_column)
            batch['news_actors'] = batch_results
            
            # Step 2: Merge back with original article metadata
            batch_with_metadata = batch.merge(
                article_metadata, 
                left_on='uri', 
                right_index=True, 
                how='left'
            )
            
            # Step 3: Expand actors to individual rows
            expanded_batch = self.expand_actors_to_rows(batch_with_metadata)
            
            # Step 4: Only save if we have actors
            if not expanded_batch.empty:
                expanded_batch.to_csv(
                    output_file_path, 
                    mode='a',
                    header=not header_written,
                    index=False, 
                    sep=';', 
                    quoting=csv.QUOTE_NONNUMERIC
                )
                header_written = True
                print(f"Batch {batch_idx + 1} processed: {len(expanded_batch)} actor rows appended")
            else:
                print(f"Batch {batch_idx + 1} contained no actors, skipping...")
        
        print(f"All batches completed! Check {output_file_path} for final actor-level results.")


def load_data(data_path: str, text_column: str = 'text', 
             title_column: str = 'title') -> pd.DataFrame:
    """Load and prepare data from CSV"""
    df = pd.read_csv(data_path, encoding='utf-8-sig', sep=';', quoting=csv.QUOTE_NONNUMERIC)
    df['uri'] = df['uri'].astype(int)
    # make all colnames lowercase
    
    df['full_text'] = df[title_column.lower()].fillna('') + '\n\n' + df[text_column.lower()].fillna('')
    df['full_text'] = df['full_text'].str.replace(',external', ' ', regex=False)
    df['full_text'] = df['full_text'].str.strip()
    df['full_text'] = df['full_text'].str.replace('\n+', '\n', regex=True)
    df['full_text'] = df['full_text'].str.replace(' +', ' ', regex=True)
        
    return df

def main(args):
    # Initialize annotator
    annotator = ActorAnnotator(
        model_name=args.model_name,
        seed=args.seed,
        gpu=args.gpu
    )
    
    # Load data
    try:
        print(f"Loading data from {args.data_path}...")
        df = load_data(args.data_path, args.text_column, args.title_column)
        print(f"Loaded {len(df)} articles")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    # Process all batches 
    print(f"Starting batch processing with batch size {args.batch_size}...")
    annotator.process_all_batches(
        df=df,
        batch_size=args.batch_size,
        output_file_path=args.output_file_path,
        text_column='full_text'  
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process actor annotations")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help="Path to input CSV data file")
    parser.add_argument('--text_column', type=str, default='text', help="Column name containing article text")
    parser.add_argument('--title_column', type=str, default='title', help="Column name containing article title")
    parser.add_argument('--output_file_path', type=str, required=True, help="Path to output CSV file")
    parser.add_argument('--batch_size', type=int, default=10, help="Number of articles to process per batch")
    parser.add_argument('--model_name', type=str, default='gpt-oss:20b', help="Ollama model name")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument('--gpu', type=int, default=0, help="GPU index to use")

    args = parser.parse_args()
    main(args)