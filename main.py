#main.py

import os
import logging
import re
from typing import List, Optional, Set, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel, Field
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Preprocessing ---
def preprocess_text_backend(text):
    if not isinstance(text, str) or not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if len(word) > 2 and not word.isdigit()])
    return text

# --- Load Environment Variables & Initialize Supabase ---
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.critical("Supabase URL or Key not found. Application cannot start.")
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI()

# --- Add CORS middleware ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load TF-IDF Model ---
try:
    vectorizer: TfidfVectorizer = joblib.load('tfidf_vectorizer.joblib')
    product_ids_for_matrix_all: List[int] = joblib.load('product_ids_for_matrix.joblib')
    logger.info("TF-IDF vectorizer and product IDs loaded successfully.")
except FileNotFoundError:
    logger.critical("TF-IDF model files not found. Run train_tfidf_model.py first.")
    raise RuntimeError("TF-IDF model files not found. Run train_tfidf_model.py first.")
except Exception as e:
    logger.critical(f"Critical error loading TF-IDF model files: {e}")
    raise RuntimeError(f"Error loading TF-IDF model files: {e}")

# --- Pydantic Models ---
class RecommendedProductModel(BaseModel):
    # Python field names are snake_case
    product_id: int = Field(..., alias="Product_Id") # Alias for input from DB-like dicts
    name: str = Field(..., alias="Product_Name")
    brand: Optional[str] = Field(None, alias="Brand")
    price: Optional[float] = Field(None, alias="Price")
    image_url: Optional[str] = Field(None, alias="Image_Url")
    description: Optional[str] = Field(None, alias="Product_Description")
    product_type: str = Field(..., alias="Product_Type")
    similarity_score: Optional[float] = None

    class Config:
        populate_by_name = True # Allows Pydantic to use aliases when populating model from dict

class RoutineStepModel(BaseModel):
    step_name: str
    product_type_expected: str
    recommended_products: List[RecommendedProductModel] = []

class SkincareRoutineResponse(BaseModel):
    morning_routine: List[RoutineStepModel]
    night_routine: List[RoutineStepModel]

class UserProfileInput(BaseModel):
    skin_type_id: int
    sensitivity: str
    concern_ids: Set[int] = Field(default_factory=set)

# --- Mappings & Configs ---
SKIN_TYPE_NAME_TO_DB_COLUMN: Dict[str, str] = {
    "Oily": "Oily", "Dry": "Dry", "Combination": "Combination",
    "Sensitive": "Sensitive", "Normal": "Normal", "Mature": "Mature",
    "Acne-prone": "Acne-prone"
}
ROUTINE_STEPS_CONFIG = {
    "morning": [
        {'step_name': 'Cleanser', 'product_type': 'Cleanser'},
        {'step_name': 'Toner', 'product_type': 'Toner'},
        {'step_name': 'Serum', 'product_type': 'Serum'},
        {'step_name': 'Moisturizer', 'product_type': 'Moisturizer'},
        {'step_name': 'Sunscreen', 'product_type': 'Sunscreen'},
    ],
    "night": [
        {'step_name': 'Cleanser', 'product_type': 'Cleanser'},
        {'step_name': 'Toner', 'product_type': 'Toner'},
        {'step_name': 'Treatment', 'product_type': 'Treatment'},
        {'step_name': 'Moisturizer', 'product_type': 'Moisturizer'},
        {'step_name': 'Eye Cream', 'product_type': 'Eye cream'},
    ],
}

# --- Helper Functions ---
async def get_skin_type_name_from_id(skin_type_id: int) -> Optional[str]:
    # ... (implementation from previous correct version)
    try:
        res = supabase.table("Skin Types").select("skin_type").eq("skin_type_id", skin_type_id).limit(1).execute()
        return res.data[0]['skin_type'] if res.data else None
    except Exception as e:
        logger.error(f"Error fetching skin type name for ID {skin_type_id}: {e}")
        return None

async def get_concern_details_from_ids(concern_ids: Set[int]) -> List[dict]:
    # ... (implementation from previous correct version)
    if not concern_ids: return []
    try:
        int_concern_ids = [int(cid) for cid in concern_ids if str(cid).isdigit()]
        if not int_concern_ids: return []
        res = supabase.table("Skin Concerns").select("concern_id, concern").in_("concern_id", int_concern_ids).execute()
        return res.data if res.data else []
    except Exception as e:
        logger.error(f"Error fetching concern details for IDs {concern_ids}: {e}")
        return []

def generate_tfidf_query_string(
    product_type_for_step: str, user_selected_skin_type_name: Optional[str],
    is_user_sensitive_input: bool, user_selected_concern_details: List[dict]
) -> str:
    # ... (implementation from previous correct version - the comprehensive one)
    query_parts_set = {preprocess_text_backend(product_type_for_step)}
    if user_selected_skin_type_name:
        st_lower = user_selected_skin_type_name.lower()
        query_parts_set.add(preprocess_text_backend(st_lower))
        if st_lower == "oily": query_parts_set.update(preprocess_text_backend(s) for s in ['oil control', 'lightweight', 'non comedogenic', 'mattifying', 'pore minimizing', 'gel cleanser', 'water based', 'oily skin solutions'])
        elif st_lower == "dry": query_parts_set.update(preprocess_text_backend(s) for s in ['hydrating', 'nourishing', 'rich moisture', 'ceramides', 'hyaluronic acid', 'emollient', 'cream', 'balm for dry skin', 'intense hydration'])
        elif st_lower == "combination": query_parts_set.update(preprocess_text_backend(s) for s in ['balancing', 'combination skin care', 't zone control', 'light hydration', 'non greasy formula', 'regulate sebum'])
        elif st_lower == "sensitive": query_parts_set.update(preprocess_text_backend(s) for s in ['sensitive skin', 'gentle formula', 'soothing care', 'calming ingredients', 'hypoallergenic tested', 'fragrance free products', 'minimalist skincare', 'non irritating', 'for reactive skin', 'barrier repair for sensitive skin'])
        elif st_lower == "normal": query_parts_set.update(preprocess_text_backend(s) for s in ['normal skin type', 'maintaining balance', 'healthy skin glow', 'daily essentials', 'all skin types suitable', 'gentle maintenance'])
        elif st_lower == "mature": query_parts_set.update(preprocess_text_backend(s) for s in ['mature skin care', 'anti aging benefits', 'age defense system', 'firming and lifting', 'collagen boosting', 'retinol alternatives', 'wrinkle reduction'])
        elif st_lower == "acne-prone": query_parts_set.update(preprocess_text_backend(s) for s in ['acne prone skin treatment', 'blemish control formula', 'breakout prevention', 'salicylic acid based', 'tea tree oil products', 'clarifying solutions', 'non comedogenic for acne'])
    if is_user_sensitive_input and user_selected_skin_type_name != "Sensitive": query_parts_set.update(preprocess_text_backend(s) for s in ['gentle', 'soothing', 'calming', 'for sensitive users', 'non irritating formulation'])
    for detail in user_selected_concern_details:
        concern_name_raw = str(detail.get('concern', ''))
        concern_name_processed = preprocess_text_backend(concern_name_raw)
        if concern_name_processed:
            query_parts_set.add(concern_name_processed)
            cn_lower_raw = concern_name_raw.lower()
            if "acne" in cn_lower_raw: query_parts_set.update(preprocess_text_backend(s) for s in ['blemish spot', 'pimple remedy', 'anti acne ingredients'])
            if "post blemish scar" in cn_lower_raw or "scar" in cn_lower_raw: query_parts_set.update(preprocess_text_backend(s) for s in ['scar fading', 'hyperpigmentation after acne', 'skin repair', 'mark reduction', 'evening tone'])
            if "redness" in cn_lower_raw: query_parts_set.update(preprocess_text_backend(s) for s in ['anti redness serum', 'rosacea care', 'centella asiatica cica', 'azelaic acid cream', 'reduce skin redness', 'calm irritation'])
            if "enlarged pores" in cn_lower_raw or "pores" in cn_lower_raw: query_parts_set.update(preprocess_text_backend(s) for s in ['pore minimizing toner', 'refine pores treatment', 'tighten pores solution', 'smooth skin texture for pores'])
            if "impaired skin barrier" in cn_lower_raw or "barrier" in cn_lower_raw: query_parts_set.update(preprocess_text_backend(s) for s in ['skin barrier repair cream', 'strengthen skin barrier', 'ceramide rich', 'protective skincare', 'restore barrier function'])
            if "uneven skin tone" in cn_lower_raw or "uneven tone" in cn_lower_raw: query_parts_set.update(preprocess_text_backend(s) for s in ['even skin tone serum', 'brightening complex', 'correct discoloration', 'tone correcting moisturizer'])
            if "texture" in cn_lower_raw: query_parts_set.update(preprocess_text_backend(s) for s in ['smoothing serum', 'resurfacing treatment', 'exfoliate gently', 'improve skin texture', 'aha bha peel'])
            if "radiance" in cn_lower_raw or "dullness" in cn_lower_raw: query_parts_set.update(preprocess_text_backend(s) for s in ['radiance boosting serum', 'glowing skin products', 'exfoliating toner for dullness', 'revitalizing mask', 'illuminate complexion', 'vitamin c for radiance'])
            if "elasticity" in cn_lower_raw or "aging skin" in cn_lower_raw: query_parts_set.update(preprocess_text_backend(s) for s in ['improve skin elasticity', 'firming lotion', 'anti aging support', 'collagen peptides', 'retinol for aging', 'youthful appearance'])
            if "blackheads" in cn_lower_raw: query_parts_set.update(preprocess_text_backend(s) for s in ['blackhead removal', 'pore clearing', 'salicylic acid for blackheads', 'deep cleanse pores'])
            if "hyperpigmentation" in cn_lower_raw or "dark spots" in cn_lower_raw: query_parts_set.update(preprocess_text_backend(s) for s in ['dark spot corrector', 'reduce hyperpigmentation', 'melanin inhibitor', 'brighten dark spots'])
            if "dryness and dehydration" in cn_lower_raw or "dehydration" in cn_lower_raw: query_parts_set.update(preprocess_text_backend(s) for s in ['intense hydration for dry skin', 'dehydrated skin relief', 'hyaluronic acid booster', 'glycerin moisturizer', 'quench thirsty skin', 'plumping serum'])
            if "dark circles" in cn_lower_raw: query_parts_set.update(preprocess_text_backend(s) for s in ['dark circle eye cream', 'brighten under eye', 'vitamin k eye treatment', 'reduce eye shadows'])
            if "puffiness" in cn_lower_raw: query_parts_set.update(preprocess_text_backend(s) for s in ['de puffing eye gel', 'reduce eye bags', 'caffeine eye serum', 'cooling eye treatment'])
    final_query_parts = {part for part in query_parts_set if part}
    logger.info(f"Generated TF-IDF Query for {product_type_for_step}: {' '.join(list(final_query_parts))}")
    return " ".join(list(final_query_parts))


async def recommend_products_for_step_backend(
    product_type_for_step: str, user_selected_skin_type_name: Optional[str],
    is_user_sensitive_input: bool, all_user_concern_details: List[dict]
) -> List[RecommendedProductModel]:
    logger.info(f"Recommending: Step='{product_type_for_step}', SkinType='{user_selected_skin_type_name}', Sensitive='{is_user_sensitive_input}', Concerns='{[c.get('concern') for c in all_user_concern_details]}'")
    db_skin_type_column_for_filter = SKIN_TYPE_NAME_TO_DB_COLUMN.get(user_selected_skin_type_name or "")
    query = supabase.table("Products").select("Product_Id, Product_Name, Brand, Price, Image_Url, Product_Description, Product_Type")
    query = query.eq("Product_Type", product_type_for_step)
    if db_skin_type_column_for_filter: query = query.eq(db_skin_type_column_for_filter, 1)
    if is_user_sensitive_input: query = query.eq("Sensitive", 1)
    concern_db_columns_to_filter = [SKIN_TYPE_NAME_TO_DB_COLUMN[detail.get('concern')] for detail in all_user_concern_details if detail.get('concern') in SKIN_TYPE_NAME_TO_DB_COLUMN]
    for concern_col in set(concern_db_columns_to_filter): query = query.eq(concern_col, 1)
    try:
        candidate_products_raw = query.execute().data
        if candidate_products_raw is None: candidate_products_raw = []
        logger.info(f"Found {len(candidate_products_raw)} rule-based candidates for {product_type_for_step}.")
    except Exception as e:
        logger.error(f"Error fetching candidates for {product_type_for_step}: {e}")
        return []
    if not candidate_products_raw: return []
    try:
        tfidf_matrix_all_products = joblib.load('tfidf_matrix.joblib')
    except Exception as e:
        logger.error(f"Error loading tfidf_matrix.joblib: {e}. Falling back to price sort.")
        candidate_products_raw.sort(key=lambda p: p.get("Price") or float('inf'))
        # Ensure data passed to model_validate matches Pydantic field names or aliases
        return [RecommendedProductModel.model_validate(p_data) for p_data in candidate_products_raw[:5] if p_data]

    candidate_product_vectors, valid_candidate_products_for_ranking = [], []
    product_id_to_matrix_idx = {pid: i for i, pid in enumerate(product_ids_for_matrix_all)}
    for prod_data in candidate_products_raw:
        prod_id = prod_data.get("Product_Id")
        if prod_id is not None and prod_id in product_id_to_matrix_idx:
            matrix_idx = product_id_to_matrix_idx[prod_id]
            candidate_product_vectors.append(tfidf_matrix_all_products[matrix_idx])
            valid_candidate_products_for_ranking.append(prod_data) # This still has DB keys (PascalCase)
    if not valid_candidate_products_for_ranking:
        logger.info(f"No valid candidates for TF-IDF ranking for {product_type_for_step}.")
        candidate_products_raw.sort(key=lambda p: p.get("Price") or float('inf'))
        return [RecommendedProductModel.model_validate(p_data) for p_data in candidate_products_raw[:5] if p_data]

    tfidf_query_text = generate_tfidf_query_string(product_type_for_step, user_selected_skin_type_name, is_user_sensitive_input, all_user_concern_details)
    processed_query = preprocess_text_backend(tfidf_query_text)
    if not processed_query:
        logger.info(f"Processed TF-IDF query is empty for {product_type_for_step}. Using rule-based matches.")
        valid_candidate_products_for_ranking.sort(key=lambda p: p.get("Price") or float('inf'))
        return [RecommendedProductModel.model_validate(p_data) for p_data in valid_candidate_products_for_ranking[:5] if p_data]
    query_vector = vectorizer.transform([processed_query])
    if not candidate_product_vectors: return []
    try:
        candidate_matrix_sparse = vstack(candidate_product_vectors)
        similarities = cosine_similarity(query_vector, candidate_matrix_sparse)
    except Exception as e:
        logger.error(f"Error during TF-IDF similarity calculation: {e}. Falling back to price sort.")
        valid_candidate_products_for_ranking.sort(key=lambda p: p.get("Price") or float('inf'))
        return [RecommendedProductModel.model_validate(p_data) for p_data in valid_candidate_products_for_ranking[:5] if p_data]

    ranked_products_with_scores = []
    for i, prod_data in enumerate(valid_candidate_products_for_ranking):
        score = 0.0
        if similarities.shape[1] > i: score = float(similarities[0, i])
        ranked_products_with_scores.append({**prod_data, "_similarityScore": score}) # prod_data still has DB keys
    ranked_products_with_scores.sort(key=lambda p: (p["_similarityScore"], -(p.get("Price") or float('-inf'))), reverse=True)

    result_models = []
    for p_data_from_ranking in ranked_products_with_scores[:5]:
        dict_for_pydantic = p_data_from_ranking.copy()
        dict_for_pydantic['similarity_score'] = dict_for_pydantic.pop('_similarityScore', None)

        # Check the value that will be used for the 'product_id' field by Pydantic,
        # considering the alias "Product_Id"
        product_id_value = dict_for_pydantic.get("Product_Id") # Key from database/raw data

        if product_id_value is None:
            logger.warning(f"Skipping product with NULL 'Product_Id' (database key). Name: {dict_for_pydantic.get('Product_Name')}, Raw Data: {p_data_from_ranking}")
            continue
        try:
            # Pydantic will use the alias "Product_Id" from dict_for_pydantic
            # to populate the internal 'product_id' (snake_case) field of the model.
            validated_model = RecommendedProductModel.model_validate(dict_for_pydantic)
            result_models.append(validated_model)
        except Exception as val_err:
            logger.error(f"Pydantic validation error for product {dict_for_pydantic.get('Product_Name')}: {val_err}. Data: {dict_for_pydantic}")
            
    logger.info(f"Recommended {len(result_models)} products for {product_type_for_step} after TF-IDF.")
    return result_models

# --- API Endpoint ---
@app.post("/build_routine", response_model=SkincareRoutineResponse)
async def build_routine_endpoint(profile: UserProfileInput):
    logger.info(f"Received /build_routine request: {profile.model_dump_json(indent=2)}")
    user_selected_skin_type_name = await get_skin_type_name_from_id(profile.skin_type_id)
    if not user_selected_skin_type_name:
        logger.warning(f"Invalid skin_type_id provided: {profile.skin_type_id}")
        raise HTTPException(status_code=400, detail=f"Invalid skin_type_id: {profile.skin_type_id}")
    is_user_sensitive_input = profile.sensitivity.lower() == "yes"
    user_selected_concern_details = await get_concern_details_from_ids(profile.concern_ids)
    logger.info(f"User Profile Parsed: Type='{user_selected_skin_type_name}', Sensitive={is_user_sensitive_input}, Concerns Count={len(user_selected_concern_details)}")
    
    morning_routine_steps, night_routine_steps = [], []
    for step_config in ROUTINE_STEPS_CONFIG["morning"]:
        products = await recommend_products_for_step_backend(
            step_config['product_type'], user_selected_skin_type_name,
            is_user_sensitive_input, user_selected_concern_details
        )
        morning_routine_steps.append(RoutineStepModel(step_name=step_config['step_name'], product_type_expected=step_config['product_type'], recommended_products=products))
    for step_config in ROUTINE_STEPS_CONFIG["night"]:
        products = await recommend_products_for_step_backend(
            step_config['product_type'], user_selected_skin_type_name,
            is_user_sensitive_input, user_selected_concern_details
        )
        night_routine_steps.append(RoutineStepModel(step_name=step_config['step_name'], product_type_expected=step_config['product_type'], recommended_products=products))
    
    response_pydantic_model = SkincareRoutineResponse(morning_routine=morning_routine_steps, night_routine=night_routine_steps)
    
    # When FastAPI serializes `response_pydantic_model`, it will call model_dump() on it.
    # By default, model_dump() (and subsequently model_dump_json()) uses FIELD NAMES, not aliases.
    # So the output JSON should have "product_id", "name", etc. (snake_case)
    # which matches what your Dart client expects.
    
    json_to_log = response_pydantic_model.model_dump_json(indent=2, by_alias=False) # Explicitly by_alias=False
    
    logger.info(f"--- Backend sending response (length: {len(json_to_log)}) ---")
    logger.info(json_to_log)
    logger.info(f"--- End of backend JSON response ---")
    
    return response_pydantic_model.model_dump(by_alias=False) # <--- CRITICAL CHANGE # FastAPI handles serialization based on model config

@app.get("/")
async def read_root():
    return {"message": "Skincare Routine Builder API is running. Access /docs for API details."}

# To run: uvicorn main:app --reload --host 0.0.0.0 --port 8000