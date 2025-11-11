import pandas as pd
import json
import csv
import re
import os
import sys
import ollama
from ollama import Client
import argparse
import time
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

class ActorAnnotator:
    def __init__(self, model_name: str = "gpt-oss:20b", seed: int = 0, 
                 ollama_host: str = "http://localhost:11434", timeout: int = 120):
        self.model_name = model_name
        self.seed = seed
        self.system_prompt = None
        self.ollama_host = ollama_host
        self.timeout = timeout  # ADD THIS LINE
        self.main_prompt = None
        self._model_checked = False
        self._load_prompts()
        self.client = Client(host=ollama_host, timeout=timeout)

        # Validate prompts were loaded successfully
        if not self.system_prompt or not self.main_prompt:
            raise ValueError("Failed to load required prompts. Check if actor_extraction_prompt.txt exists and is properly formatted.")

    def _recreate_client(self):
        """Recreate the Ollama client to prevent connection issues"""
        try:
            # Try to close existing client if it has a close method
            if hasattr(self.client, 'close'):
                self.client.close()
        except:
            pass
        
        # Delete old client
        del self.client
        
        # Create new client with timeout
        self.client = Client(host=self.ollama_host, timeout=self.timeout)  # ADD timeout parameter
        print("Ollama client recreated")

    def _restart_ollama_server(self):
        """Restart Ollama server to clear server-side memory"""
        import subprocess
        print("Restarting Ollama server...")
        try:
            # First stop Ollama
            subprocess.run(["pkill", "-9", "ollama"], check=False)
            time.sleep(2)
            
            # Re-start Ollama in background
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(5)  
            print("Ollama server restarted")
        except Exception as e:
            print(f"Failed to restart Ollama: {e}")

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
            
            model_names = []
            if hasattr(models_response, 'models'):
                for model in models_response.models:
                    if hasattr(model, 'model'):
                        model_names.append(model.model)
            
            # Pull model if not found
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
    
    def extract_actors_from_content(self, news_content: str, ollama_model: str = None, 
                                   timeout_seconds: int = None) -> Tuple[str, str]:
        """Extract raw LLM response from news content using Ollama.
        Uses Ollama's built-in timeout, restarts client after timeout.
        Has a pre-set num_ctx value to handle longer articles, can be adapted if needed.
        Timeout is handled at the client level. 120 seconds suffice.
        """
        if ollama_model is None:
            ollama_model = self.model_name
        
        # Use instance timeout if not specified
        if timeout_seconds is None:
            timeout_seconds = self.timeout

        if not news_content or len(news_content.strip()) < 50:
            return "", "" 
        
        if not self.system_prompt or not self.main_prompt:
            print("Prompts not loaded properly")
            return "", ""  
        
        try:
            prompt = self.build_prompt(
                system_prompt=self.system_prompt, 
                main_prompt=self.main_prompt + "\n" + news_content
            )

            response = self.client.chat(
                model=ollama_model, 
                messages=prompt, 
                options={
                    'temperature': 0.0, 
                    'seed': self.seed,
                    'num_ctx': 6144  
                }
            )
            
            response_content = response['message']['content'].strip()
            
            if response_content:
                cleaned_response = self.clean_and_parse_actors(response_content)
                return response_content, cleaned_response
            
            # Empty response (successful call but no content)
            return "", ""
            
        except TimeoutError:
            print(f"Request timed out after {timeout_seconds}s - skipping article and restarting client")
            self._recreate_client()
            return "", ""
        
        except Exception as e:
            error_msg = str(e).lower()
            # Check if it's a connection-related error
            if any(keyword in error_msg for keyword in ['connection', 'timeout', 'broken pipe', 'reset', 'refused']):
                print(f"Connection error: {e} - restarting client")
                self._recreate_client()
            else:
                print(f"Error during extraction: {e}")
            return "", ""

    def clean_and_parse_actors(self, llm_response: str) -> Optional[str]:
        """Clean LLM response and extract structured actor data."""

        cleaned = re.sub(r"\s+", " ", llm_response)
        cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)

        try:
            actors_pattern = r'"?actors"?\s*:\s*\[(.*)\]'
            actors_match = re.search(actors_pattern, cleaned, re.DOTALL | re.IGNORECASE)
            
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
                actors_list = self._extract_actor_json_from_output(actors_content)
            
            if actors_list:
                # Check if actors use standard field names
                has_standard_fields = any(
                    'actor_name' in actor 
                    for actor in actors_list 
                    if isinstance(actor, dict)
                )
                
                # Check if actors use alternative field names
                alternative_field_names = ['name', 'actor', 'role', 'function', 'type', 'party']
                has_alternative_fields = any(
                    any(field in actor for field in alternative_field_names)
                    for actor in actors_list 
                    if isinstance(actor, dict)
                )
                
                # If no standard fields but has alternatives, it's a field name issue
                if not has_standard_fields and has_alternative_fields:
                    print("Actors use alternative field names - trying fallback")
                    return self._fallback_clean_and_parse_actors(llm_response)
                
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
            
            # No Fallback Trigger 3 - empty after filtering is legitimate
            if not cleaned_actors:
                print("No valid actors after filtering (legitimate empty)")
                return None
            
            return json.dumps({'actors': cleaned_actors})
            
        except Exception as e:
            print(f"Unexpected exception during parsing: {e}")
            if len(llm_response.strip()) > 20:  # Has substantial content
                print("Trying fallback due to exception")
                try: 
                    fallback_result = self._fallback_clean_and_parse_actors(llm_response)
                    if fallback_result:
                        return fallback_result
                except Exception as fe:
                    print(f"Fallback parser also failed: {fe}")
            return None
            
    def _fallback_llm_output_cleaner(self, llm_response: str) -> Optional[str]:
        """
        Improved fallback parser that handles various LLM output formats.
        Handles:
        - Bare arrays without "actors" wrapper
        - Many field name variations (name/actor, role/function/actor_type, party/pp)
        - Missing "actors" key
        - Inconsistent JSON formatting
        - CamelCase field names (actorName, actorType, actorParty)
        - Fields with spaces ("actor name", "name/description")
        - Array values for single fields (actor_type: ["a"])
        """
        cleaned = re.sub(r"\s+", " ", llm_response)
        cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        
        try:
            # Try to parse as JSON directly
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON structures
                data = self._extract_json_from_text(cleaned)
            
            if not data:
                return None
            
            # Extract actors list from various formats
            actors_list = self._get_actors_from_data(data)
            
            if not actors_list or not isinstance(actors_list, list):
                return None
            
            # Normalize each actor dictionary
            normalized_actors = []
            for actor in actors_list:
                if isinstance(actor, dict):
                    norm_actor = self._normalize_actor_fields(actor)
                    if norm_actor and norm_actor['actor_name'].strip():
                        normalized_actors.append(norm_actor)
            
            if not normalized_actors:
                return None
            
            return json.dumps({'actors': normalized_actors})
            
        except Exception as e:
            print(f"Fallback parser failed: {e}")
            return None  
        
    def _extract_json_from_text(self, text: str):
        """Extract JSON structure from messy text."""
        # Try to find object with actors key
        obj_pattern = r'\{[^{}]*"actors"\s*:\s*\[[^\]]*\][^{}]*\}'
        obj_match = re.search(obj_pattern, text, re.DOTALL | re.IGNORECASE)
        if obj_match:
            try:
                return json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try to find bare array of objects
        array_pattern = r'\[\s*\{[^]]+\}\s*(?:,\s*\{[^]]+\}\s*)*\]'
        array_match = re.search(array_pattern, text, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None

    def _get_actors_from_data(self, data):
        """Extract actors list from various data structures."""
        # If data is already a list (bare array), return it
        if isinstance(data, list):
            return data
        
        # If data is a dict, look for actors key (case-insensitive)
        if isinstance(data, dict):
            for key in data.keys():
                if key.lower() == 'actors':
                    return data[key]
        
        return None

    def _normalize_actor_fields(self, actor: dict) -> Optional[dict]:
        """
        Normalize actor dictionary to use standard field names.
        Maps alternative field names to: actor_name, actor_function, actor_pp
        
        Handles variations like:
        - actor_name, name, actor, actorName, "actor name", "name/description", actor_name/description
        - actor_function, function, role, actor_type, actorType, category, type
        - actor_pp, party, pp, actorParty
        """
        # Define field name mappings (alternatives -> standard)
        name_variations = [
            'actor_name', 'name', 'actor', 'actorName', 
            'actor name', 'name/description', 'actor_name/description'
        ]
        function_variations = [
            'actor_function', 'function', 'role', 'actor_type', 
            'actorType', 'category', 'type', 'actor_function', 'actor_type (function)',
            'actor_f'
        ]
        party_variations = [
            'actor_pp', 'party', 'pp', 'actorParty'
        ]
        
        # Extract actor_name (required)
        actor_name = None
        for field in name_variations:
            if field in actor:
                value = actor[field]
                # Handle array values (take first element)
                if isinstance(value, list):
                    value = value[0] if value else None
                if value and value is not None:
                    actor_name = str(value).strip()
                    if actor_name:
                        break
        
        if not actor_name:
            return None
        
        # Extract actor_function (optional)
        actor_function = ''
        for field in function_variations:
            if field in actor:
                value = actor[field]
                # Handle array values (take first element or join)
                if isinstance(value, list):
                    value = value[0] if value else ''
                if value and value is not None:
                    actor_function = str(value).strip()
                    break
        
        # Extract actor_pp (optional)
        actor_pp = ''
        for field in party_variations:
            if field in actor:
                value = actor[field]
                # Handle array values
                if isinstance(value, list):
                    value = value[0] if value else ''
                # Skip null/None values
                if value is not None and value != 'null':
                    actor_pp = str(value).strip()
                    break
        
        return {
            'actor_name': actor_name,
            'actor_function': actor_function,
            'actor_pp': actor_pp
        }

    def _extract_actor_json_from_output(self, actors_text: str) -> List[Dict[str, str]]:
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
            actors = self._extract_actor_info_from_text(actors_text)
        
        return actors

    def _extract_actor_info_from_text(self, actors_text: str) -> List[Dict[str, str]]:
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

    def process_dataframe(self, 
                          df: pd.DataFrame, 
                          text_column: str, 
                          id_column: str, 
                          title_column: str,
                          timeout_seconds: int) -> pd.DataFrame:
        """
        Process a DataFrame and return it with actors column added.
        For in-memory processing without file output.
        """
        if not self._ensure_model_available():
            print("Model not available, exiting.")
            return df
        
        df['full_text'] = (df[title_column].fillna('').astype(str) + '\n' + 
                        df[text_column].fillna('').astype(str)) 
        df['full_text'] = df['full_text'].str.strip()
        df['full_text'] = df['full_text'].str.replace('\n+', '\n', regex=True)
        df['full_text'] = df['full_text'].str.replace(' +', ' ', regex=True)

        # Drop unnecessary columns to save memory
        columns_to_keep = [id_column, 'full_text']
        df_result = df[columns_to_keep].copy()
        
        print(f"Processing {len(df_result)} articles...")
        
        # Process each article
        results = []
        cleaned_results = []
        for idx, row in tqdm(df_result.iterrows(), total=len(df_result), desc="Extracting actors"):
            uri = row.get(id_column, idx)
            text = df_result.at[idx, 'full_text']
            actors_json, cleaned_response = self.extract_actors_from_content(text, timeout_seconds=timeout_seconds)

            results.append(actors_json)
            cleaned_results.append(cleaned_response)
            
            # Brief pause 
            time.sleep(0.1)
        
        df_result['news_actors'] = results
        df_result['raw_response'] = cleaned_results
        return df_result   


    def chunk_dataframe(self, df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
        """Split dataframe into chunks for batch processing"""
        return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    def process_articles_batch(self, df_batch: pd.DataFrame, text_column: str, timeout: int = 120) -> Tuple[List[str], List[str]]:
        """Process a batch of articles and return raw and cleaned results.
        
        Returns:
            Tuple[List[str], List[str]]: (raw_outputs, cleaned_outputs)
        """
        raw_output = []
        cleaned_output = []
        
        for idx, row in df_batch.iterrows():
            text = row.get(text_column, '')
            raw_response, cleaned_response = self.extract_actors_from_content(text, timeout_seconds=timeout)  
            raw_output.append(raw_response)
            cleaned_output.append(cleaned_response)
        
        return raw_output, cleaned_output


    def process_all_batches(self, df: pd.DataFrame, batch_size: int, 
                    output_file_path: str, text_column: str, id_column: str,
                    timeout: int = 120, start_batch: int = 0) -> None:
        """Process all batches with memory management"""
        import gc
        import psutil
        
        batches = self.chunk_dataframe(df, batch_size)
        
        # Check if output file exists to determine if we should write header
        header_written = os.path.exists(output_file_path)

        print(f"\n{'='*60}")
        print(f"STARTING BATCH PROCESSING")
        print(f"{'='*60}")
        print(f"Total articles: {len(df)}")
        print(f"Batch size: {batch_size}")
        print(f"Total batches: {len(batches)}")
        print(f"Starting from batch: {start_batch + 1}")  # +1 for human-readable
        print(f"Batches to process: {len(batches) - start_batch}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Skip to start_batch
        for batch_idx in range(start_batch, len(batches)):
            batch = batches[batch_idx]
            batch_start = time.time()
            
            # Monitor memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            print(f"\n--- Batch {batch_idx + 1}/{len(batches)} ({len(batch)} articles) ---")
            print(f"Memory: {memory_mb:.2f} MB")
            
            # Recreate client every 50 batches
            if batch_idx > 0 and batch_idx % 50 == 0:
                print("⚙ Recreating client...")
                self._recreate_client()
            
            batch = batch.reset_index(drop=True)
            
            # Process batch
            batch_results, cleaned_results = self.process_articles_batch(batch, text_column, timeout=timeout)
            
            batch['news_actors_raw'] = batch_results
            batch['news_actors'] = cleaned_results
        
            # Write results
            with open(output_file_path, 'a', encoding='utf-8') as f:
                batch.to_csv(
                    f, 
                    header=not header_written,
                    index=False, 
                    sep=';', 
                    quoting=csv.QUOTE_NONNUMERIC
                )
            header_written = True
            
            batch_time = time.time() - batch_start
            print(f"✓ Batch completed in {batch_time:.1f}s")
            
            # Cleanup
            del batch
            del batch_results
            del cleaned_results
            gc.collect()
        
        total_time = time.time() - start_time
        avg_per_article = total_time / len(df) if len(df) > 0 else 0
    
        print(f"\n{'='*60}")
        print(f"✓ ALL BATCHES COMPLETED!")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f}s)")
        print(f"Average per article: {avg_per_article:.1f}s")
        print(f"Total articles processed: {len(df)}")
        print(f"Output saved to: {output_file_path}")
        print(f"{'='*60}\n")


def load_data(data_path: str, text_column: str, title_column: str, id_column: str) -> pd.DataFrame:
    """Load and prepare data from CSV"""
        
    # Validate required columns exist
    required_cols = [text_column, title_column, id_column]
    dtypes={col: str for col in required_cols}
    df = pd.read_csv(data_path, 
                     encoding='utf-8-sig', 
                     sep=';', 
                     quoting=csv.QUOTE_NONNUMERIC,
                     usecols=required_cols,
                    dtype=dtypes)

    # Create full_text column
    df['full_text'] = (df[title_column].fillna('').astype(str) + '\n' + 
                      df[text_column].fillna('').astype(str)) 
    df['full_text'] = df['full_text'].str.strip()
    df['full_text'] = df['full_text'].str.replace('\n+', '\n', regex=True)
    df['full_text'] = df['full_text'].str.replace(' +', ' ', regex=True)

    # Drop unnecessary columns to save memory
    columns_to_keep = [id_column, 'full_text']
    df = df[[id_column, 'full_text']]
        
    return df

def main(args):
    # Validate input file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Input file not found: {args.data_path}")
        return
    
    # Initialize annotator
    annotator = ActorAnnotator(
        model_name=args.model_name,
        seed=args.seed,
    )

    # Ensure model is available
    if not annotator._ensure_model_available():
        print("Model not available, exiting.")
        return
    
    # Load data with explicit arguments
    try:
        print(f"Loading data from {args.data_path}...")
        df = load_data(
            data_path=args.data_path, 
            text_column=args.text_column, 
            title_column=args.title_column,
            id_column=args.id_column
        )
        print(f"Loaded {len(df)} articles")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    # Process all batches with explicit arguments
    print(f"Starting batch processing with batch size {args.batch_size}...")
    annotator.process_all_batches(
        df=df,
        batch_size=args.batch_size,
        output_file_path=args.output_file_path,
        text_column='full_text',  
        id_column=args.id_column,
        timeout=args.timeout,
        start_batch=args.start_batch  
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process actor annotations")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help="Path to input CSV data file")
    parser.add_argument('--text_column', type=str, default='text', help="Column name containing article text")
    parser.add_argument('--title_column', type=str, default='title', help="Column name containing article title")
    parser.add_argument('--id_column', type=str, default='uri', help="Column name containing unique article ID")
    parser.add_argument('--output_file_path', type=str, required=True, help="Path to output CSV file")
    parser.add_argument('--batch_size', type=int, default=10, help="Number of articles to process per batch")
    parser.add_argument('--model_name', type=str, default='gpt-oss:20b', help="Ollama model name")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument('--timeout', type=int, default=120, help="Timeout in seconds per article (default: 120)")
    parser.add_argument('--start_batch', type=int, default=0, help="Batch number to start from (0-indexed)")

    args = parser.parse_args()
    main(args)
