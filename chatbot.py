"""
Chatbot RAG Avanzado para Steam
Usa embeddings pre-calculados (BERT o Word2Vec) para búsqueda más rápida
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
    
    df_reviews = pd.read_csv("datasets/df_reviews.csv", index_col=0)
    print(f"✓ {len(df_reviews)} reviews cargadas")
    
    return df_reviews


def cargar_game_data_batches(max_batches=23):
    """
    Carga game_data desde los 23 batches pickle
    Contiene: quality_score, avg_user_credibility, all_reviews_text, etc.
    """
    print(f"Cargando game_data desde batches (hasta {max_batches} batches)...")
    
    game_data_list = []
    batch_num = 0
    
    while batch_num < max_batches:
        filepath = f'datasets/game_data_batch_{batch_num}.pkl'
        if not os.path.exists(filepath):
            if batch_num == 0:
                # No existe el batch 0, probablemente no hay batches
                break
            else:
                # Puede que falten batches intermedios, continuar
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
        print(f"✓ Total: {len(game_data)} juegos de {len(game_data_list)} batches con métricas avanzadas")
        return game_data
    else:
        print("⚠️  No se encontraron batches, usando datos básicos")
        return None


def cargar_embeddings_precalculados():
    """
    Carga embeddings pre-calculados (solo Word2Vec review_vectors.npy)
    NOTA: review_vectors tiene embeddings de TODAS las reviews (7.8M)
    pero nosotros solo necesitamos los de nuestros 11k juegos
    """
    print("Buscando embeddings pre-calculados...")
    
    # Word2Vec review vectors - NO CARGAR TODO, es demasiado
    if os.path.exists('datasets/review_vectors.npy'):
        print("  ⚠️  review_vectors.npy encontrado pero contiene 7.8M embeddings")
        print("  ℹ️  Es demasiado para RAM, se crearán embeddings solo para los 11k juegos")
        return None, None, None
    
    print("  ℹ️  No se encontró review_vectors.npy")
    print("  ℹ️  Se crearán embeddings nuevos solo para los juegos (~1 minuto)")
    return None, None, None


# ============================================
# PREPARACIÓN BASE DE CONOCIMIENTO
# ============================================

def preparar_base_conocimiento_avanzada(game_data, df_reviews):
    """
    Crea documentos usando game_data (con todas las métricas calculadas)
    """
    print("Preparando base de conocimiento avanzada...")
    
    documents = []
    metadata = []
    
    for idx, row in game_data.iterrows():
        # Información básica
        title = idx  # game_title es el índice
        product_id = row.get('product_id', None)
        
        # Métricas avanzadas del notebook
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
    """
    Crea índice FAISS usando embeddings pre-calculados
    """
    print(f"Creando índice FAISS desde {embedding_type} embeddings...")
    
    dimension = embeddings.shape[1]
    
    # Usar embeddings directamente sin recalcular
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    print(f"✓ Índice creado: {index.ntotal:,} vectores de dimensión {dimension}")
    
    return index, embedding_type


def crear_embeddings_desde_documentos(documents):
    """
    Crea embeddings solo para los documentos de juegos (11k)
    Usa GPU si está disponible para acelerar
    """
    print(f"Creando embeddings para {len(documents)} documentos...")
    
    from sentence_transformers import SentenceTransformer
    
    # Usar GPU si está disponible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"  ✓ Usando GPU para crear embeddings")
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Crear embeddings en batches para no saturar memoria
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
    """
    Recupera k documentos más relevantes
    Usa el mismo tipo de embedding que el índice
    """
    # Aquí necesitamos encodear la query con el mismo método
    # Por simplicidad, vamos a usar sentence-transformers para queries nuevas
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
        self.conversation_history = []  # Memoria conversacional
        
        print("Cargando Phi-3-mini (optimizado para 8GB VRAM)...")
        
        # Detectar y configurar GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  ✓ GPU detectada: {gpu_name}")
            print(f"  ✓ VRAM disponible: {vram_gb:.1f} GB")
            
            self.device = "cuda"
            # Usar 8-bit quantization para reducir uso de memoria
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,  # Carga en 8-bit (reduce 50% memoria)
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
            print(f"  ✓ Modelo cargado en GPU con cuantización 8-bit")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct",
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32
            )
            print(f"  ✓ Modelo cargado en CPU")
        
        print("✓ Sistema listo!")
    
    def responder(self, query, k=7):
        """Responde usando RAG con métricas avanzadas y memoria conversacional"""
        # Recuperar contexto
        context_docs, context_metadata = recuperar_contexto(
            query, self.index, self.documents, self.metadata, 
            self.embedding_type, k=k
        )
        
        # Crear prompt - Formato correcto para Phi-3
        context_text = "\n\n".join([f"[Doc {i+1}]\n{doc}" for i, doc in enumerate(context_docs)])
        
        messages = [
            {
                "role": "system",
                "content": f"""Eres un experto en videojuegos de Steam. Respondes en español de manera natural y conversacional.

Tienes acceso a información detallada sobre juegos: puntuaciones de calidad, reseñas de usuarios, géneros, precios y más.

Información disponible:
{context_text}

IMPORTANTE: Habla de forma natural como si fueras un experto que conoce estos juegos. NO menciones "documentos", "contexto" o "información proporcionada". Simplemente recomienda juegos basándote en tu conocimiento."""
            }
        ]
        
        # Añadir historial conversacional (últimos 6 mensajes para no saturar)
        messages.extend(self.conversation_history[-6:])
        
        # Añadir pregunta actual
        messages.append({
            "role": "user",
            "content": query
        })
        
        # Usar apply_chat_template para formato correcto
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generar respuesta
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
        
        # Decodificar solo los tokens nuevos (sin el prompt)
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Guardar en historial (sin el contexto largo del sistema)
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response.strip()})
        
        return response.strip(), context_metadata
    
    def limpiar_historial(self):
        """Limpia el historial conversacional"""
        self.conversation_history = []
        return "Historial conversacional limpiado"


# ============================================
# CHAINLIT HANDLERS
# ============================================

@cl.on_chat_start
async def start():
    """Se ejecuta cuando inicia el chat"""
    global chatbot_instance, index, documents, metadata, df_reviews
    
    await cl.Message(
        content="🎮 **Bienvenido al Asistente de Juegos de Steam**\n\n"
        "Soy tu experto en recomendaciones de juegos. Puedo ayudarte con:\n\n"
        "• 🎯 Recomendaciones personalizadas basadas en calidad y popularidad\n"
        "• 📊 Análisis de juegos según reviews y tiempo de juego\n"
        "• ⭐ Búsqueda de juegos por género, características o estilo\n"
        "• 💬 Opiniones de la comunidad y valoraciones de usuarios\n"
        "• 🧠 **Memoria conversacional** - Recuerdo nuestras conversaciones previas\n\n"
        "**Inicializando sistema...**"
    ).send()
    
    if chatbot_instance is None:
        try:
            msg = cl.Message(content="Preparando sistema...")
            await msg.send()
            
            # Cargar reviews
            df_reviews = cargar_reviews_csv()
            
            # Cargar game_data con métricas (los batches ya tienen toda la info)
            game_data = cargar_game_data_batches(max_batches=23)
            
            if game_data is None:
                await cl.Message(
                    content="❌ No se encontraron datos de juegos. Verifica que existan los archivos game_data_batch en datasets/"
                ).send()
                return
            
            # Preparar documentos
            documents, metadata = preparar_base_conocimiento_avanzada(game_data, df_reviews)
            
            # Cargar embeddings pre-calculados (ahora retorna None para evitar OOM)
            embeddings, embed_indices, embed_type = cargar_embeddings_precalculados()
            
            if embeddings is not None:
                # Crear índice desde embeddings pre-calculados
                index, _ = crear_indice_desde_embeddings(embeddings, documents, metadata, embed_type)
            else:
                # Crear embeddings nuevos SOLO para los 11k juegos (no 7.8M reviews)
                embeddings = crear_embeddings_desde_documentos(documents)
                
                # Crear índice FAISS
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings.astype('float32'))
                embed_type = 'sentence-transformer'
                
                # IMPORTANTE: Liberar memoria antes de cargar Phi-3
                del embeddings
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("  ℹ️  Memoria liberada antes de cargar modelo")
            
            # Inicializar chatbot
            chatbot_instance = SteamChatbotAdvanced(index, documents, metadata, embed_type)
            
            await msg.stream_token("\n✅ Sistema listo. ¡Hazme cualquier pregunta sobre juegos!")
            await msg.update()
            
        except Exception as e:
            await cl.Message(
                content=f"❌ Error al inicializar: {str(e)}"
            ).send()
            return
    else:
        await cl.Message(content="✅ Sistema listo. ¿En qué puedo ayudarte?").send()


@cl.on_message
async def main(message: cl.Message):
    """Handler de mensajes"""
    global chatbot_instance
    
    if chatbot_instance is None:
        await cl.Message(content="❌ Sistema no inicializado. Por favor espera.").send()
        return
    
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        await msg.stream_token("🔍 Buscando información...\n\n")
        
        respuesta, context_metadata = chatbot_instance.responder(message.content, k=7)
        
        await msg.stream_token(respuesta)
        
        await msg.update()
        
    except Exception as e:
        await msg.stream_token(f"\n\n❌ Error: {str(e)}")
        await msg.update()


@cl.on_chat_end
def end():
    print("Chat ended")


if __name__ == "__main__":
    pass