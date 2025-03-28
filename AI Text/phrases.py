from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import re #Ajout de la librairie re

# Charger le modèle et le tokenizer
model_path = "./gpt2_french_finetuned"  # Ou le nom du modèle pré-entraîné si tu ne l'as pas fine-tuné
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

def generate_text(input_text, max_length=50, num_return_sequences=1,
                   top_k=20, top_p=0.95, temperature=0.7, repetition_penalty=1.5, num_beams=5, no_repeat_ngram_size=3):
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    generated_text = generator(
        input_text,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping= True # Arret lorsque tout les sequences on atteint le token eos
    )

    # Supprimer le prompt de la sortie générée
    text_without_prompt = generated_text[0]['generated_text'].replace(input_text, "").strip()

    # Trouver la fin de la première phrase (., !, ?, \n)
    match = re.search(r'[.!?\n]', text_without_prompt)
    if match:
        end_of_first_sentence = match.end()
        first_sentence = text_without_prompt[:end_of_first_sentence].strip()
    else:
        # Pas de ponctuation trouvée, on retourne tout
        first_sentence = text_without_prompt


    return [{'generated_text': first_sentence}] # Retourne la première phrase


# Exemples de test
input_text1 = "chat dormir canapé"
generated_text1 = generate_text(input_text1)
print(f"Input: {input_text1}")
print(f"Generated: {generated_text1[0]['generated_text']}")