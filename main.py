RAGFLOW_API_KEY = 'ragflow-M3Yjg1MWY2Zjc0ZTExZWZhNGE4MDI0Mm'
RAGFLOW_SERVER_URI = 'http://localhost'

from ragflow_sdk import RAGFlow
import os
from bs4 import BeautifulSoup
import re
import html
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

nltk.download('stopwords')

ragflow_client = RAGFlow(RAGFLOW_API_KEY, RAGFLOW_SERVER_URI)

# Get a dataset
def get_dataset_by_name(dataset_name):
    datasets = ragflow_client.list_datasets()
    for dataset in datasets:
        if dataset.name == dataset_name:
            return dataset
    print('Dataset not found')
    return None 

def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    if isinstance(html_content, bytes):
        try:
            html_content = html_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                html_content = html_content.decode('latin-1')
            except UnicodeDecodeError:
                html_content = html_content.decode('utf-8', errors='replace')
                
    for tag in soup(['script', 'style', 'noscript', 'iframe', 'img', 'meta']):
        tag.decompose()
    
    # Replace image tags with their alt text
    for img in soup.find_all('img'):
        if img.get('alt') is not None:
            img.replace_with(soup.new_string(img['alt']))
        else:
            img.decompose()
    
    # Preserve headers
    for i in range(1, 7):
        for header in soup.find_all(f'h{i}'):
            header_text = header.get_text(strip=True)
            header.replace_with(soup.new_string(f"\n\n{'#' * i} {header_text}\n"))
    
    # Preserve paragraphs
    for p in soup.find_all('p'):
        p_text = p.get_text(strip=True)
        p.replace_with(soup.new_string(f"{p_text}\n"))
    
    for ul in soup.find_all('ul'):
        for li in ul.find_all('li'):
            li_text = li.get_text(strip=True)
            li.replace_with(soup.new_string(f"• {li_text}\n"))
    
    for ol in soup.find_all('ol'):
        for i, li in enumerate(ol.find_all('li'), 1):
            li_text = li.get_text(strip=True)
            li.replace_with(soup.new_string(f"{i}. {li_text}\n"))
    
    text = soup.get_text(separator=' ')
    
    # Clean up excessive whitespace
    text = re.sub(r'\n\s+\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Decode HTML entities
    text = html.unescape(text)
    return text.strip()

# Upload documents by iterating through the directory
def upload_documents(dataset, directory_path, is_cleaned=True):
    document_ids = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, "rb") as file:
                if is_cleaned:
                    content = file.read()
                    cleaned_content = clean_html(content)
                    
                    filename = os.path.splitext(filename)[0] + ".txt"
                    txt_path = os.path.join('./news-raw/cleaned_news', filename)
                    with open(txt_path, "w", encoding='utf-8') as txt_file:
                        txt_file.write(cleaned_content)
                    with open(txt_path, "r", encoding='utf-8') as txt_file:
                        document = {"displayed_name": filename, "blob": txt_file.read()}
                else:
                    document = {"displayed_name": filename, "blob": file.read()}
            try:
                doc = dataset.upload_documents([document])  
                document_ids.append(doc[0].id)
                print(f"Uploaded: {filename}")
            except Exception as e:
                print(f"Failed to upload {filename}: {str(e)}")
    
    if document_ids:
        # Parse documents asynchronously
        dataset.async_parse_documents(document_ids)
        print("📄 Parsing started for uploaded documents.")
    else:
        print('No documents found in the directory')
        return 

# Create or make an assistant
def get_or_create_assistant(assistant_name, dataset):
    assistants = ragflow_client.list_chats()
    for assistant in assistants:
        if assistant.name == assistant_name:
            print(f'Found assistant {assistant_name}')
            return assistant
        
    # Create the assistant if not found    
    print(f'Creating assistant {assistant_name}')
    assistant = ragflow_client.create_chat(
        name=assistant_name, dataset_ids=[dataset.id]
    )
    return assistant

def create_session(assistant, session_name):
    session = assistant.create_session(session_name)
    return session

def ask_assistant(session, question):
    print(f"Question: {question}")
    
    try:
        # Get the response (might be a generator regardless of stream setting)
        response = session.ask(question, stream=False)
        
        # Check if response is a generator
        if hasattr(response, '__iter__') and hasattr(response, '__next__'):
            print("Response is a generator. Processing chunks...")
            
            # Initialize an empty string to collect the response
            full_content = ""
            
            # Process all chunks from the generator
            for chunk in response:
                # Extract content from chunk based on its type
                if hasattr(chunk, 'content'):
                    chunk_content = chunk.content
                elif isinstance(chunk, dict) and 'content' in chunk:
                    chunk_content = chunk['content']
                elif isinstance(chunk, str):
                    chunk_content = chunk
                else:
                    # If we can't determine the content format, convert to string
                    chunk_content = str(chunk)
                
                # Print chunk content and add to full response
                print(chunk_content, end='', flush=True)
                full_content += chunk_content
            
            print("\nFinished processing generator.")
            return full_content
        
        # Handle non-generator responses
        elif hasattr(response, 'content'):
            print(response.content)
            return response.content
        elif isinstance(response, dict):
            if 'content' in response:
                print(response['content'])
                return response['content']
            else:
                print(str(response))
                return str(response)
        else:
            print(str(response))
            return str(response)
            
    except Exception as e:
        print(f"Failed to get response: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

# Read text file
def read_text_file(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        return file.read()

def split_into_sentences(text):
    return sent_tokenize(text)

def create_chunks_into_sentences(sentences, target_chunk_size=500, overlap_sentences=2):
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    for sentence in sentences:
        size = len(sentence)
        if current_chunk_size + size > target_chunk_size and current_chunk:
            text = ' '.join(current_chunk)
            chunks.append(text)
            overlapped_length = min(overlap_sentences, len(current_chunk))
            current_chunk = current_chunk[-overlapped_length:] if overlapped_length > 0 else []
            current_chunk_size = sum(len(s) for s in current_chunk) + len(current_chunk) + 1
        current_chunk.append(sentence)
        current_chunk_size += size + 1 # Add 1 for space
    if current_chunk:
        text = ' '.join(current_chunk)
        chunks.append(text)
    return chunks

def extract_auto_keywords(text, num_keywords=5):
    words = nltk.word_tokenize(text.lower())
    # Remove common English stopwords
    stopwords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                   "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                   'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
                   'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                   'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
                   'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 
                   'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
                   'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
                   'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
                   'about', 'against', 'between', 'into', 'through', 'during', 'before', 
                   'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 
                   'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
                   'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 
                   'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
                   'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
                   'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 
                   'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
                   "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 
                   'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', 
                   "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
                   'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', 
                   "won't", 'wouldn', "wouldn't"])
    
    filtered_words = [word for word in words if word.isalnum() and word not in stopwords]
    
    # Calculate word frequency
    freq_dist = nltk.FreqDist(filtered_words)
    
    # Get the most common words
    keywords = [word for word, freq in freq_dist.most_common(num_keywords)]
    
    return keywords

def process_text_files_rag(input_path, output_path, index, chunk_size=500, overlap_sentences=2):
    text = read_text_file(input_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sentences = split_into_sentences(text)
    chunks = create_chunks_into_sentences(sentences, chunk_size, overlap_sentences)
    chunk_files = []
    for i, chunk in enumerate(chunks):
        output_file = os.path.join(output_path, f"chunk_{i+1:03d}_{index:03d}.txt")
        with open(output_file, 'w', encoding='utf-8') as file:
            keywords = extract_auto_keywords(chunk)
            keywords_str = ', '.join(keywords)
            file.write(f"Keywords: {keywords_str}\n\n{chunk}")
        chunk_files.append(output_file)
    return chunk_files

def process_all_files(input, output, chunk_size = 500, overlap_sentences = 2):
    os.makedirs(output, exist_ok=True)
    all_files = os.listdir(input)
    text_files = [f for f in all_files if f.lower().endswith('.txt')]
    print(all_files)
    all_chunk_files = []
    for i, filename in enumerate(text_files):
        input_file = os.path.join(input, filename)
        chunk_files = process_text_files_rag(
            input_path=input_file,
            output_path=output,
            index=i,
            chunk_size=chunk_size,
            overlap_sentences=overlap_sentences
        )
        
        all_chunk_files.extend(chunk_files)
    
    print(f"Processing complete. Created {len(all_chunk_files)} total chunks.")
    return all_chunk_files

def upload_chunks_to_ragflow(dataset_name, chunk_files):
    # Get or create dataset
    dataset = get_dataset_by_name(dataset_name)
    if not dataset:
        print(f"Creating new dataset: {dataset_name}")
        dataset = ragflow_client.create_dataset(dataset_name)
    
    # Upload chunks
    document_ids = []
    for file_path in chunk_files:
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read()
            document = {
                "displayed_name": os.path.basename(file_path),
                "blob": content
            }
        try:
            doc = dataset.upload_documents([document])
            document_ids.append(doc[0].id)
            print(f"Uploaded: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Failed to upload {file_path}: {str(e)}")
    
    if document_ids:
        # Parse documents asynchronously
        dataset.async_parse_documents(document_ids)
        print("📄 Parsing started for uploaded documents.")
    
    return document_ids

if __name__ == '__main__':
    cleaned_dataset_name = 'cleaned'
    uncleaned_dataset_name = 'sample-1'

    # cleaned_dataset = get_dataset_by_name(cleaned_dataset_name)
    # upload_documents(cleaned_dataset, './news-raw/news', is_cleaned=True)
    # uncleaned_dataset = get_dataset_by_name(uncleaned_dataset_name)
    
    # assistant_cleaned = get_or_create_assistant('sample_2_cleaned', cleaned_dataset)
    # assistant_uncleaned = get_or_create_assistant('sample-1', uncleaned_dataset)
    
    # # Ask the assistant
    # question = "How did Apple contribute to the growth of S&P 500 in 2023?"
    
    # cleaned_session = create_session(assistant_cleaned, 'cleaned_session')
    # uncleaned_session = create_session(assistant_uncleaned, 'uncleaned_session')
    
    # response_cleaned = ask_assistant(cleaned_session, question)
    # response_uncleaned = ask_assistant(uncleaned_session, question)
    
    # print("\nGetting response from CLEANED dataset:")
    # response_cleaned = ask_assistant(cleaned_session, question)
    
    # print("\nGetting response from UNCLEANED dataset:")
    # response_uncleaned = ask_assistant(uncleaned_session, question)
    
    # with open("cleaned_response.txt", "w", encoding="utf-8") as f:
    #     f.write(response_cleaned)
    
    # with open("uncleaned_response.txt", "w", encoding="utf-8") as f:
    #     f.write(response_uncleaned)
    
    input_directory = "news-raw/cleaned_news"   # Directory with source .txt files
    output_directory = "news-raw/chunks"  # Directory to save chunks
    dataset_name = "chunked_nltk"
    
    all_chunks = process_all_files(
        input=input_directory,
        output=output_directory,
        chunk_size=500,
        overlap_sentences=2
    )
    
    document_ids = upload_chunks_to_ragflow(dataset_name, all_chunks)
