"""
Chatbot RAG Avanzado para Steam
Usa embeddings pre-calculados y predicciones de Random Forest
"""

import chainlit as cl
import pandas as pd
import numpy as np
import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN GLOBAL
# ============================================

chatbot_instance = None
index = None
documents = None
metadata = None
df_reviews = None

# ============================================
# CARGA DE DATOS Y EMBEDDINGS
# ============================================

def cargar_reviews_csv():
    """Carga solo df_reviews (df_games no es necesario, ya tenemos game_data)"""
    print("Cargando reviews desde CSV...")
    
    if os.path.exists("datasets/df_reviews.csv"):
        df_reviews = pd.read_csv("datasets/df_reviews.csv", index_col=0)
        print(f"✓ {len(df_reviews)} reviews cargadas")
        return df_reviews
    else:
        print("⚠️ No se encontró datasets/df_reviews.csv")
        return pd.DataFrame()


def cargar_game_data_batches(max_batches=23):
    """
    Carga game_data desde los batches pickle.
    """
    print(f"Cargando game_data desde batches (hasta {max_batches} batches)...")
    
    game_data_list = []
    batch_num = 0
    
    while batch_num < max_batches:
        filepath = f'datasets/game_data_batch_{batch_num}.pkl'
        if not os.path.exists(filepath):
            if batch_num == 0:
                break
            else:
                batch_num += 1
                continue
        
        try:
            batch_df = pd.read_pickle(filepath)
            game_data_list.append(batch_df)
            print(f"  ✓ Batch {batch_num}: {len(batch_df)} juegos")
        except Exception as e:
            print(f"  ⚠️  Error cargando batch {batch_num}: {e}")
        
        batch_num += 1
    
    if game_data_list:
        game_data = pd.concat(game_data_list, ignore_index=False)
        print(f"✓ Total: {len(game_data)} juegos cargados inicialmente")
        return game_data
    else:
        print("⚠️  No se encontraron batches, usando datos básicos")
        return None

def recalcular_categoria(score):
    """Recalcula la categoría de texto basada en el nuevo score numérico"""
    if score >= 95: return "Overwhelmingly Positive"
    if score >= 80: return "Very Positive"
    if score >= 70: return "Mostly Positive"
    if score >= 40: return "Mixed"
    if score >= 20: return "Mostly Negative"
    return "Negative"

def integrar_predicciones_rf(game_data):
    """
    Carga datasets/predicciones_rf_test.csv y actualiza los quality_score
    en el dataframe principal game_data.
    """
    path_predicciones = "datasets/predicciones_rf_test.csv"
    print(f"Integrando predicciones RF desde {path_predicciones}...")
    
    if not os.path.exists(path_predicciones):
        print("⚠️ No se encontró el archivo de predicciones. Se usarán los scores originales.")
        return game_data

    try:
        # Cargar CSV
        df_pred = pd.read_csv(path_predicciones)
        
        # Intentar detectar columnas clave automáticamente
        # Buscamos columnas que puedan contener el nombre del juego (index) y el score
        col_id = None
        col_score = None
        
        # Posibles nombres para la columna de ID (Nombre del juego)
        candidates_id = ['app_name', 'title', 'game_title', 'name']
        for c in candidates_id:
            if c in df_pred.columns:
                col_id = c
                break
        
        # Posibles nombres para la columna de Score (Predicción)
        candidates_score = ['predicted_quality_score', 'prediction', 'quality_score', 'y_pred', 'score']
        for c in candidates_score:
            if c in df_pred.columns:
                col_score = c
                break
        
        # Fallbacks si no se encuentran nombres obvios
        if col_id is None: col_id = df_pred.columns[0] # Asumir primera columna es ID
        if col_score is None: col_score = df_pred.select_dtypes(include=np.number).columns[-1] # Asumir última numérica es score
        
        print(f"  ℹ️ Mapeando: ID='{col_id}' -> Score='{col_score}'")

        # Preparar dataframe de predicciones para el merge
        df_pred = df_pred.set_index(col_id)
        
        # Eliminar duplicados si existen en el CSV de predicciones
        df_pred = df_pred[~df_pred.index.duplicated(keep='first')]

        # Encontrar intersección de juegos
        common_games = game_data.index.intersection(df_pred.index)
        
        if len(common_games) > 0:
            # Actualizar Quality Score solo donde hay coincidencia
            game_data.loc[common_games, 'quality_score'] = df_pred.loc[common_games, col_score]
            
            # Recalcular la categoría (texto) para mantener coherencia en el RAG
            # ya que si el score cambia de 40 a 90, la categoría no debe seguir diciendo "Mixed"
            print("  ℹ️ Recalculando categorías de texto basadas en los nuevos scores...")
            game_data.loc[common_games, 'quality_category'] = game_data.loc[common_games, 'quality_score'].apply(recalcular_categoria)
            
            print(f"✓ Predicciones actualizadas para {len(common_games)} juegos exitosamente.")
        else:
            print("⚠️ No hubo coincidencias de nombres entre game_data y el archivo de predicciones.")
            
    except Exception as e:
        print(f"❌ Error integrando predicciones: {e}")
    
    return game_data


def cargar_embeddings_precalculados():
    """
    Carga embeddings pre-calculados (solo Word2Vec review_vectors.npy)
    """
    print("Buscando embeddings pre-calculados...")
    
    if os.path.exists('datasets/review_vectors.npy'):
        print("  ⚠️  review_vectors.npy encontrado pero contiene 7.8M embeddings")
        print("  ℹ️  Es demasiado para RAM, se crearán embeddings solo para los 11k juegos")
        return None, None, None
    
    print("  ℹ️  No se encontró review_vectors.npy")
    return None, None, None


# ============================================
# PREPARACIÓN BASE DE CONOCIMIENTO
# ============================================

def preparar_base_conocimiento_avanzada(game_data, df_reviews):
    """
    Crea documentos usando game_data (con todas las métricas calculadas y actualizadas)
    """
    print("Preparando base de conocimiento avanzada...")
    
    documents = []
    metadata = []
    
    for idx, row in game_data.iterrows():
        # Información básica
        title = idx  # game_title es el índice
        product_id = row.get('product_id', None)
        
        # Métricas avanzadas (ahora con quality_score actualizado por RF)
        quality_score = row.get('quality_score', 0)
        quality_category = row.get('quality_category', 'Unknown')
        num_reviews = row.get('num_reviews', 0)
        avg_hours = row.get('avg_hours', 0)
        avg_user_credibility = row.get('avg_user_credibility', 0)
        total_useful_votes = row.get('total_useful_votes', 0)
        avg_review_length = row.get('avg_review_length', 0)
        
        # Información adicional
        price = row.get('price', 'N/A')
        genres = row.get('genres', 'Unknown')
        tags = row.get('tags', '')
        metascore = row.get('metascore', 'N/A')
        
        # Texto de reviews concatenado
        all_reviews_text = row.get('all_reviews_text', '')[:800]
        
        # Crear documento rico
        doc = f"""Game: {title}
Quality Score: {quality_score:.1f}/100 (Category: {quality_category})
Number of Reviews: {int(num_reviews)}
Average Playtime: {avg_hours:.1f} hours
User Credibility: {avg_user_credibility:.2f}
Total Useful Votes: {int(total_useful_votes)}
Average Review Length: {avg_review_length:.1f} chars
Price: {price}
Genres: {genres}
Tags: {tags}
Metascore: {metascore}
Reviews Summary: {all_reviews_text}"""
        
        documents.append(doc)
        metadata.append({
            'product_id': product_id,
            'title': title,
            'quality_score': quality_score,
            'quality_category': quality_category,
            'num_reviews': int(num_reviews),
            'avg_hours': avg_hours,
            'avg_user_credibility': avg_user_credibility,
            'price': price,
            'genres': genres
        })
    
    print(f"✓ {len(documents)} documentos creados con métricas completas")
    return documents, metadata


def crear_indice_desde_embeddings(embeddings, documents, metadata, embedding_type):
    """Crea índice FAISS usando embeddings pre-calculados"""
    print(f"Creando índice FAISS desde {embedding_type} embeddings...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    print(f"✓ Índice creado: {index.ntotal:,} vectores de dimensión {dimension}")
    return index, embedding_type


def crear_embeddings_desde_documentos(documents):
    """Crea embeddings solo para los documentos de juegos (11k)"""
    print(f"Creando embeddings para {len(documents)} documentos...")
    from sentence_transformers import SentenceTransformer
    
    # Usar GPU si está disponible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"  ✓ Usando GPU para crear embeddings")
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    batch_size = 64 if device == 'cuda' else 32
    embeddings = embedder.encode(
        documents, 
        batch_size=batch_size,
        show_progress_bar=True, 
        convert_to_numpy=True,
        device=device
    )
    
    print(f"✓ Embeddings creados: {len(embeddings)} vectores de dimensión {embeddings.shape[1]}")
    return embeddings


def recuperar_contexto(query, index, documents, metadata, embedding_type, k=7):
    """Recupera k documentos más relevantes"""
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    retrieved_docs = [documents[idx] for idx in indices[0] if idx < len(documents)]
    retrieved_metadata = [metadata[idx] for idx in indices[0] if idx < len(metadata)]
    
    return retrieved_docs, retrieved_metadata


# ============================================
# CLASE CHATBOT
# ============================================

class SteamChatbotAdvanced:
    def __init__(self, index, documents, metadata, embedding_type):
        self.index = index
        self.documents = documents
        self.metadata = metadata
        self.embedding_type = embedding_type
        self.conversation_history = [] 
        
        print("Cargando Phi-3-mini (optimizado para 8GB VRAM)...")
        
        if torch.cuda.is_available():
            self.device = "cuda"
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            print("  ℹ️  Usando cuantización 8-bit para ahorrar VRAM")
        else:
            print("  ⚠️  GPU no detectada, usando CPU")
            self.device = "cpu"
            quantization_config = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            trust_remote_code=True
        )
        
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct",
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct",
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32
            )
        
        print("✓ Sistema listo!")
    
    def responder(self, query, k=7):
        context_docs, context_metadata = recuperar_contexto(
            query, self.index, self.documents, self.metadata, 
            self.embedding_type, k=k
        )
        
        context_text = "\n\n".join([f"[Doc {i+1}]\n{doc}" for i, doc in enumerate(context_docs)])
        
        messages = [
            {
                "role": "system",
                "content": f"""Eres un experto en videojuegos de Steam. Respondes en español de manera natural y conversacional.

Tienes acceso a información detallada sobre juegos, incluyendo Puntuaciones de Calidad (Quality Score) que provienen de un análisis avanzado de Inteligencia Artificial (Random Forest).

Información disponible:
{context_text}

IMPORTANTE: Habla de forma natural como si fueras un experto. Usa los datos para fundamentar tus recomendaciones. Si un juego tiene un score alto, destácalo."""
            }
        ]
        
        messages.extend(self.conversation_history[-6:])
        messages.append({"role": "user", "content": query})
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000)
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=600,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response.strip()})
        
        return response.strip(), context_metadata


# ============================================
# CHAINLIT HANDLERS
# ============================================

@cl.on_chat_start
async def start():
    """Se ejecuta cuando inicia el chat"""
    global chatbot_instance, index, documents, metadata, df_reviews
    
    await cl.Message(
        content="🎮 **Bienvenido al Asistente de Juegos de Steam (AI Enhanced)**\n\n"
        "Soy tu experto en recomendaciones. Uso un modelo de **Random Forest** para predecir la calidad real de los juegos.\n\n"
        "**Inicializando sistema y cargando predicciones...**"
    ).send()
    
    if chatbot_instance is None:
        try:
            msg = cl.Message(content="Cargando datos...")
            await msg.send()
            
            # 1. Cargar reviews
            df_reviews = cargar_reviews_csv()
            
            # 2. Cargar game_data base
            game_data = cargar_game_data_batches(max_batches=23)
            
            if game_data is None:
                await cl.Message(content="❌ Error: No se encontraron datos de juegos.").send()
                return
            
            # 3. INTEGRACIÓN DE PREDICCIONES (NUEVO)
            # Aquí es donde inyectamos los scores del CSV de Random Forest
            game_data = integrar_predicciones_rf(game_data)
            
            # 4. Preparar documentos (ya usarán los nuevos scores)
            documents, metadata = preparar_base_conocimiento_avanzada(game_data, df_reviews)
            
            # 5. Embeddings y Modelo (Lógica estándar)
            embeddings, _, embed_type = cargar_embeddings_precalculados()
            
            if embeddings is not None:
                index, _ = crear_indice_desde_embeddings(embeddings, documents, metadata, embed_type)
            else:
                embeddings = crear_embeddings_desde_documentos(documents)
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings.astype('float32'))
                embed_type = 'sentence-transformer'
                
                del embeddings
                import gc
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            chatbot_instance = SteamChatbotAdvanced(index, documents, metadata, embed_type)
            
            await msg.stream_token("\n✅ Sistema listo con predicciones de IA cargadas.")
            await msg.update()
            
        except Exception as e:
            await cl.Message(content=f"❌ Error al inicializar: {str(e)}").send()
            return
    else:
        await cl.Message(content="✅ Sistema listo.").send()


@cl.on_message
async def main(message: cl.Message):
    global chatbot_instance
    if chatbot_instance is None:
        await cl.Message(content="❌ Sistema no inicializado.").send()
        return
    
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        await msg.stream_token("🔍 Analizando con IA...\n\n")
        respuesta, _ = chatbot_instance.responder(message.content, k=7)
        await msg.stream_token(respuesta)
        await msg.update()
        
    except Exception as e:
        await msg.stream_token(f"\n\n❌ Error: {str(e)}")
        await msg.update()