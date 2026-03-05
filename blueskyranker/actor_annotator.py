import polars as pl
import json
import re
import os
import ollama
from ollama import Client
import time
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import unicodedata


class ActorAnnotator:
    def __init__(self, model_name: str = "gpt-oss:20b",
                 seed: int = 42,
                 ollama_host: str = "http://localhost:11434",
                 timeout: int = 120):

        self.model_name = model_name
        self.seed = seed
        self.system_prompt = None
        self.ollama_host = ollama_host
        self.timeout = timeout
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
            if hasattr(self.client, 'close'):
                self.client.close()
        except:
            pass
        del self.client
        self.client = Client(host=self.ollama_host, timeout=self.timeout)
        print("Ollama client recreated")

    def _restart_ollama_server(self):
        """Restart Ollama server to clear server-side memory"""
        import subprocess
        print("Restarting Ollama server...")
        try:
            subprocess.run(["pkill", "-9", "ollama"], check=False)
            time.sleep(2)
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
            print(f"Checking availability of model: {self.model_name}")
            models_response = ollama.list()
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

    def extract_actors_from_content(self, 
                                    news_content: str, 
                                    ollama_model: str = None) -> Tuple[str, str]:
        """Extract raw LLM response from news content using Ollama.
        Uses Ollama's built-in timeout through client, restarts client after timeout.
        Has a pre-set num_ctx value to handle longer articles, can be adapted if needed.
        """
        if ollama_model is None:
            ollama_model = self.model_name

        # cutting too short articles to prevent unnecessary LLM calls
        if not news_content or len(news_content.strip()) < 50:
            return "", ""

        # cutting too long articles to prevent memory issues
        MAX_CHARS = 12000
        if len(news_content) > MAX_CHARS:
            news_content = news_content[:MAX_CHARS]

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
                    'num_ctx': 6144,
                    'num_gpu': 999  # use all layers of the model
                }
            )

            response_content = response['message']['content'].strip()

            if response_content:
                cleaned_response = self.clean_and_parse_actors(response_content)
                return response_content, cleaned_response
            
            return "", ""

        except Exception as e:
            error_msg = str(e).lower()
            is_timeout = isinstance(e, TimeoutError) or any(kw in error_msg for kw in ['timed out', 'timeout'])
            is_connection = any(kw in error_msg for kw in ['connection', 'broken pipe', 'reset', 'refused'])

            if is_timeout:
                print(f"Request timed out after {self.timeout}s - restarting client")
                words = news_content.split()
                print(f"Failed article has {len(words)} words")
                print(f"First 200 characters of content: {news_content[:200]}")
                self._recreate_client()
                # retry with longer timeout
                try:
                    extended_client = Client(host=self.ollama_host, timeout=self.timeout * 2)
                    response = extended_client.chat(
                        model=ollama_model,
                        messages=prompt,
                        options={
                            'temperature': 0.0,
                            'seed': self.seed,
                            'num_ctx': 6144,
                            'num_gpu': 999
                        }
                    )
                    response_content = response['message']['content'].strip()
                    if response_content:
                        cleaned_response = self.clean_and_parse_actors(response_content)
                        return response_content, cleaned_response
                except Exception as retry_e:
                    print(f"Retry after timeout also failed: {retry_e}")
            elif is_connection:
                print(f"Connection error: {e} - restarting client")
                self._recreate_client()
            else:
                print(f"Error during extraction: {e}")
            return "", ""

    def _normalize_text(self, text: str) -> str:
        """Normalize text by handling special characters and encoding issues."""
        if not text:
            return text
        
        text = unicodedata.normalize('NFC', text)
        text = ' '.join(text.split())
        
        return text.strip()

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
                actors_list = json.loads(f"[{actors_content}]")
            except json.JSONDecodeError:
                print("Direct JSON parsing failed, trying pattern extraction")
                actors_list = self._extract_actor_json_from_output(actors_content)

            if actors_list:
                has_standard_fields = any(
                    'actor_name' in actor for actor in actors_list if isinstance(actor, dict)
                )
                alternative_field_names = ['name', 'actor', 'role', 'function', 'type', 'party']
                has_alternative_fields = any(
                    any(field in actor for field in alternative_field_names)
                    for actor in actors_list if isinstance(actor, dict)
                )
                if not has_standard_fields and has_alternative_fields:
                    print("Actors use alternative field names - trying fallback")
                    return self._fallback_llm_output_cleaner(llm_response)

            cleaned_actors = []
            for actor in actors_list:
                if isinstance(actor, dict) and 'actor_name' in actor:
                    cleaned_actor = {
                        'actor_name': self._normalize_text(str(actor.get('actor_name', '')).strip()),
                        'actor_function': self._normalize_text(str(actor.get('actor_function', '')).strip()),
                        'actor_pp': self._normalize_text(str(actor.get('actor_pp', '')).strip())
                    }
                    if cleaned_actor['actor_name']:
                        cleaned_actors.append(cleaned_actor)

            if not cleaned_actors:
                print("No valid actors after filtering (legitimate empty)")
                return None
            
            return json.dumps({'actors': cleaned_actors}, ensure_ascii=False)

        except Exception as e:
            print(f"Unexpected exception during parsing: {e}")
            if len(llm_response.strip()) > 20:
                print("Trying fallback due to exception")
                try:
                    fallback_result = self._fallback_llm_output_cleaner(llm_response)
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
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                data = self._extract_json_from_text(cleaned)
            if not data:
                return None
            
            actors_list = self._get_actors_from_data(data)

            if not actors_list or not isinstance(actors_list, list):
                return None
            
            normalized_actors = []
            for actor in actors_list:
                if isinstance(actor, dict):
                    norm_actor = self._normalize_actor_fields(actor)
                    if norm_actor and norm_actor['actor_name'].strip():
                        normalized_actors.append(norm_actor)

            if not normalized_actors:
                return None
            
            return json.dumps({'actors': normalized_actors}, ensure_ascii=False)
        
        except Exception as e:
            print(f"Fallback parser failed: {e}")
            return None

    def _extract_json_from_text(self, text: str):
        """Extract JSON structure from messy text."""
        obj_pattern = r'\{[^{}]*"actors"\s*:\s*\[[^\]]*\][^{}]*\}'
        obj_match = re.search(obj_pattern, text, re.DOTALL | re.IGNORECASE)
        if obj_match:
            try:
                return json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                pass

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
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in data.keys():
                if key.lower() == 'actors':
                    return data[key]
        return None

    def _normalize_actor_fields(self, actor: dict) -> Optional[dict]:
        """
        Normalize actor dictionary to use standard field names.
        Maps alternative field names to: actor_name, actor_function, actor_pp
        """
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

        actor_name = None
        for field in name_variations:
            if field in actor:
                value = actor[field]
                if isinstance(value, list):
                    value = value[0] if value else None
                if value and value is not None:
                    actor_name = str(value).strip()
                    if actor_name:
                        break

        if not actor_name:
            return None

        actor_function = ''
        for field in function_variations:
            if field in actor:
                value = actor[field]
                if isinstance(value, list):
                    value = value[0] if value else ''
                if value and value is not None:
                    actor_function = str(value).strip()
                    break

        actor_pp = ''
        for field in party_variations:
            if field in actor:
                value = actor[field]
                if isinstance(value, list):
                    value = value[0] if value else ''
                if value is not None and value != 'null':
                    actor_pp = str(value).strip()
                    break

        return {
            'actor_name': self._normalize_text(actor_name),
            'actor_function': self._normalize_text(actor_function),
            'actor_pp': self._normalize_text(actor_pp)
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
                          df: pl.DataFrame,
                          text_column: str,
                          id_column: str,
                          title_column: str) -> pl.DataFrame:
        """
        Process a Polars DataFrame and return it with actors columns added.
        Receives the Polars DataFrame produced by load_posts_df() from the fetcher.

        Returns a Polars DataFrame with columns:
            [id_column, 'full_text', 'news_actors', 'raw_response']
        """
        if not self._ensure_model_available():
            print("Model not available, exiting.")
            return df

        # Build full_text: title + newline + article text, then clean whitespace
        full_text_series = (
            df[title_column].fill_null('').cast(pl.Utf8) + '\n' +
            df[text_column].fill_null('').cast(pl.Utf8)
        ).str.strip_chars().str.replace_all(r'\n+', '\n').str.replace_all(r' +', ' ')

        df_result = df.select([id_column]).with_columns(full_text_series.alias('full_text'))

        print(f"Processing {len(df_result)} articles...")

        texts = df_result['full_text'].to_list()
        results: List[str] = []
        cleaned_results: List[str] = []

        for text in tqdm(texts, desc="Extracting actors"):
            actors_json, cleaned_response = self.extract_actors_from_content(text or '')
            results.append(actors_json)
            cleaned_results.append(cleaned_response)
            time.sleep(0.1)

        return df_result.with_columns([
            pl.Series('news_actors', results),
            pl.Series('raw_response', cleaned_results),
        ])

    def chunk_dataframe(self, df: pl.DataFrame, chunk_size: int) -> List[pl.DataFrame]:
        """Split dataframe into chunks for batch processing"""
        return [df.slice(i, chunk_size) for i in range(0, len(df), chunk_size)]

    def process_articles_batch(self, df_batch: pl.DataFrame, text_column: str) -> Tuple[List[str], List[str]]:
        """Process a batch of articles and return raw and cleaned results.

        Returns:
            Tuple[List[str], List[str]]: (raw_outputs, cleaned_outputs)
        """
        raw_output: List[str] = []
        cleaned_output: List[str] = []

        for row in df_batch.iter_rows(named=True):
            text = row.get(text_column) or ''
            raw_response, cleaned_response = self.extract_actors_from_content(text)
            raw_output.append(raw_response)
            cleaned_output.append(cleaned_response)

        return raw_output, cleaned_output

    def process_all_batches(self,
                            df: pl.DataFrame,
                            batch_size: int,
                            text_column: str = "news_content", 
                            start_batch: int = 0,
                            output_dir: str = "actor_annotation_batches",
                            run_id: Optional[str] = None,
                            title_column: Optional[str] = "news_title") -> Tuple[pl.DataFrame, str]:
        """Process all batches with memory management, return annotated Polars DataFrame.

        Saves each completed batch immediately to disk as Parquet, mirroring the
        push_exports pattern in pipeline.py. A crash mid-run loses only the
        in-progress batch; completed batches are safe on disk.

        Args:
            df: Input DataFrame with all pipeline columns.
            batch_size: Number of articles per batch.
            text_column: Column containing article body text.
            start_batch: Resume from this batch index (0-based). Pass the run_id
                returned by a previous interrupted call to resume correctly.
            output_dir: Parent directory for batch files (e.g. "actor_annotation_batches").
            run_id: Timestamp string identifying this run (e.g. "20260303T142301").
                Auto-generated when None. Pass back the run_id from a previous call
                to resume into the same directory.
            title_column: If provided, prepend title to body text before LLM inference,
                matching process_dataframe() behaviour.

        Returns:
            Tuple of (result_df, run_id). result_df contains all original columns
            plus 'news_actors_raw' and 'news_actors'. run_id identifies the batch
            directory for resumption if needed.
        """
        import gc
        import psutil
        from pathlib import Path
        from datetime import datetime

        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%dT%H%M%S")

        run_dir = Path(output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        batches = self.chunk_dataframe(df, batch_size)

        print(f"\n{'='*60}")
        print(f"STARTING BATCH PROCESSING")
        print(f"{'='*60}")
        print(f"Total articles: {len(df)}")
        print(f"Batch size: {batch_size}")
        print(f"Total batches: {len(batches)}")
        print(f"Starting from batch: {start_batch + 1}")
        print(f"Batches to process: {len(batches) - start_batch}")
        print(f"Run directory: {run_dir}")
        print(f"{'='*60}\n")

        start_time = time.time()
        effective_text_col = '_full_text' if title_column else text_column

        for batch_idx in range(start_batch, len(batches)):
            batch = batches[batch_idx]
            batch_start = time.time()

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            print(f"\n--- Batch {batch_idx + 1}/{len(batches)} ({len(batch)} articles) ---")
            print(f"Memory: {memory_mb:.2f} MB")

            if title_column:
                full_text_series = (
                    batch[title_column].fill_null('').cast(pl.Utf8) + '\n' +
                    batch[text_column].fill_null('').cast(pl.Utf8)
                ).str.strip_chars().str.replace_all(r'\n+', '\n').str.replace_all(r' +', ' ')
                batch = batch.with_columns(full_text_series.alias('_full_text'))

            batch_results, cleaned_results = self.process_articles_batch(batch, effective_text_col)

            if title_column and '_full_text' in batch.columns:
                batch = batch.drop('_full_text')

            batch = batch.with_columns([
                pl.Series('news_actors_raw', batch_results),
                pl.Series('news_actors', cleaned_results),
            ])

            batch_path = run_dir / f"batch_{batch_idx:04d}.parquet"
            batch.write_parquet(batch_path)

            batch_time = time.time() - batch_start
            print(f"✓ Batch completed in {batch_time:.1f}s — saved to {batch_path}")

            del batch_results
            del cleaned_results
            del batch
            gc.collect()

        total_time = time.time() - start_time
        avg_per_article = total_time / len(df) if len(df) > 0 else 0

        print(f"\n{'='*60}")
        print(f"✓ ALL ACTOR ANNOTATION BATCHES COMPLETED!")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f}s)")
        print(f"Average per article: {avg_per_article:.1f}s")
        print(f"Total articles processed: {len(df)}")
        print(f"Run directory: {run_dir}")
        print(f"{'='*60}\n")

        saved_files = sorted(run_dir.glob("batch_*.parquet"))
        if not saved_files:
            return pl.DataFrame(), run_id

        result = pl.concat([pl.read_parquet(f) for f in saved_files])
        return result, run_id

