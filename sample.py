import json
import numpy as np
from bert_score import score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load data
with open('sample.json', 'r') as file:
    data = json.load(file)

# Function to compute BERT embeddings
def get_bert_embeddings(texts, model_name="bert-base-uncased"):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Generate embeddings
    embeddings = []
    
    for text in texts:
        if not text:  # Skip empty texts
            embeddings.append(None)
            continue
            
        # Tokenize and prepare input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model output
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use the [CLS] token embedding as the sentence embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        embeddings.append(embedding)
    
    return embeddings

# Function to calculate BERT Score
def calculate_bert_scores(candidates, references):
    scores = []
    for i in range(len(candidates)):
        if not candidates[i] or not references[i]:
            scores.append(None)
            continue
            
        P, R, F1 = score([candidates[i]], [references[i]], lang="en", verbose=False)
        scores.append({"Precision": P.item(), "Recall": R.item(), "F1": F1.item()})
    return scores

def analyze_bert_scores():
    # Extract cleaned and answer texts
    cleaned_texts = [item["Wrong"] for item in data]
    uncleaned_texts = [item["Right"] for item in data]
    answer_texts = [item["Answer"] for item in data]
    questions = [item["Question"] for item in data]
    
    # Get BERT embeddings
    cleaned_embeddings = get_bert_embeddings(cleaned_texts)
    uncleaned_embeddings = get_bert_embeddings(uncleaned_texts)
    answer_embeddings = get_bert_embeddings(answer_texts)
    
    # Calculate BERT scores
    cleaned_bert_scores = calculate_bert_scores(answer_texts, cleaned_texts)
    uncleaned_bert_scores = calculate_bert_scores(answer_texts, uncleaned_texts)
    
    # Calculate cosine similarities between embeddings
    cleaned_cosine_sims = []
    for i in range(len(cleaned_embeddings)):
        if cleaned_embeddings[i] is None or answer_embeddings[i] is None:
            cleaned_cosine_sims.append(None)
        else:
            sim = cosine_similarity([cleaned_embeddings[i]], [answer_embeddings[i]])[0][0]
            cleaned_cosine_sims.append(sim)
    
    uncleaned_cosine_sims = []
    for i in range(len(uncleaned_embeddings)):
        if uncleaned_embeddings[i] is None or answer_embeddings[i] is None:
            uncleaned_cosine_sims.append(None)
        else:
            sim = cosine_similarity([uncleaned_embeddings[i]], [answer_embeddings[i]])[0][0]
            uncleaned_cosine_sims.append(sim)
        
    # Compile results
    results = []
    for i in range(len(questions)):
        result = {
            "Question": questions[i],
            "Answer": answer_texts[i],
            "Cleaned_BERTScore_F1": cleaned_bert_scores[i]["F1"] if cleaned_bert_scores[i] else None,
            "Cleaned_BERTScore_Precision": cleaned_bert_scores[i]["Precision"] if cleaned_bert_scores[i] else None,
            "Cleaned_BERTScore_Recall": cleaned_bert_scores[i]["Recall"] if cleaned_bert_scores[i] else None,
            "Cleaned_Cosine_Similarity": cleaned_cosine_sims[i],
            "Uncleaned_BERTScore_F1": uncleaned_bert_scores[i]["F1"] if uncleaned_bert_scores[i] else None,
            "Uncleaned_BERTScore_Precision": uncleaned_bert_scores[i]["Precision"] if uncleaned_bert_scores[i] else None,
            "Uncleaned_BERTScore_Recall": uncleaned_bert_scores[i]["Recall"] if uncleaned_bert_scores[i] else None,
            "Uncleaned_Cosine_Similarity": uncleaned_cosine_sims[i],
        }
        results.append(result)
    
    # Create a DataFrame
    df = pd.DataFrame(results)
    
    # Display results
    print("\nBERT Score Analysis Results:")
    print(df[["Question", "Cleaned_BERTScore_F1", 'Uncleaned_BERTScore_F1']])
    
    # Calculate average scores
    cleaned_avg_f1 = np.mean([score for score in df["Cleaned_BERTScore_F1"] if score is not None])
    uncleaned_avg_f1 = np.mean([score for score in df["Uncleaned_BERTScore_F1"] if score is not None])
    cleaned_avg_cosine = np.mean([sim for sim in df["Cleaned_Cosine_Similarity"] if sim is not None])
    uncleaned_avg_cosine = np.mean([sim for sim in df["Uncleaned_Cosine_Similarity"] if sim is not None])

    
    print(f"\nAverage Wrong BERT Score F1: {cleaned_avg_f1:.4f}")
    print(f"Average Right BERT Score F1: {uncleaned_avg_f1:.4f}")
    print(f"Average Wrong Cosine Similarity: {cleaned_avg_cosine:.4f}")
    print(f"Average Right Cosine Similarity: {uncleaned_avg_cosine:.4f}")
    
    # Print comparison
    print("\nDifference (Wrong - Right):")
    print(f"BERT Score F1 Difference: {(cleaned_avg_f1 - uncleaned_avg_f1):.4f}")
    print(f"Cosine Similarity Difference: {(cleaned_avg_cosine - uncleaned_avg_cosine):.4f}")
    
    # Visualize results - Comparison between Wrong and Right
    plt.figure(figsize=(14, 10))
    
    # BERT Score F1 comparison plot
    plt.subplot(2, 1, 1)
    
    # Group data for plotting
    plot_data = []
    for i, row in df.iterrows():
        if row["Cleaned_BERTScore_F1"] is not None and row["Uncleaned_BERTScore_F1"] is not None:
            answer_text = row["Answer"][:20] + "..." if row["Answer"] else "N/A"
            plot_data.append({"Answer": answer_text, "Score": row["Cleaned_BERTScore_F1"], "Type": "Wrong"})
            plot_data.append({"Answer": answer_text, "Score": row["Uncleaned_BERTScore_F1"], "Type": "Right"})
    
    plot_df = pd.DataFrame(plot_data)
    if not plot_df.empty:
        sns.barplot(x="Answer", y="Score", hue="Type", data=plot_df)
        plt.ylabel("BERT Score F1")
        plt.title("Comparison of BERT Score F1: Wrong vs. Right")
        plt.xticks(rotation=45)
        plt.legend(title="Text Type")
    
    # Cosine similarity comparison plot
    plt.subplot(2, 1, 2)
    
    # Group data for plotting
    plot_data = []
    for i, row in df.iterrows():
        if row["Cleaned_Cosine_Similarity"] is not None and row["Uncleaned_Cosine_Similarity"] is not None:
            answer_text = row["Answer"][:20] + "..." if row["Answer"] else "N/A"
            plot_data.append({"Answer": answer_text, "Similarity": row["Cleaned_Cosine_Similarity"], "Type": "Wrong"})
            plot_data.append({"Answer": answer_text, "Similarity": row["Uncleaned_Cosine_Similarity"], "Type": "Right"})
    
    plot_df = pd.DataFrame(plot_data)
    if not plot_df.empty:
        sns.barplot(x="Answer", y="Similarity", hue="Type", data=plot_df)
        plt.ylabel("Cosine Similarity")
        plt.title("Comparison of Cosine Similarity: Wrong vs. Right")
        plt.xticks(rotation=45)
        plt.legend(title="Text Type")
    
    plt.tight_layout()
    plt.savefig("bert_score_comparison.png")
    
    return df

# Run the analysis
results_df = analyze_bert_scores()

# Save results to CSV
results_df.to_csv("sample_bert_score.csv", index=False)

print("\nAnalysis complete. Results saved to 'bert_score_results.csv'")