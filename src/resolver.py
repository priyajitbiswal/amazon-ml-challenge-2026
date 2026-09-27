"""
Global Candidate Exclusivity & Address Consistency Resolver (Phase 6)
Amazon ML Challenge 2026: Business Entity Resolution

Enforces the ground-truth invariant:
- Each candidate record (S2/S3) belongs to AT MOST ONE Source 1 entity.
- Conflicting multi-assigned candidates are resolved to the single S1 entity with
  highest street and building number affinity.
- Prunes false cross-street merges and rescues unpolluted singletons.
"""

import re
import unicodedata
from typing import Dict, List, Set, Tuple, Optional
import duckdb
import pyarrow as pa
import pandas as pd
from rapidfuzz.distance import Levenshtein

STREET_TYPES = {
    'rue', 'r', 'avenue', 'av', 'ave', 'boulevard', 'bd', 'blvd', 'bvd',
    'chemin', 'che', 'chem', 'allee', 'all', 'impasse', 'imp', 'route', 'rte',
    'place', 'pl', 'quai', 'cours', 'cour', 'square', 'sq', 'passage', 'pass',
    'rond-point', 'esplanade', 'voie', 'street', 'st', 'road', 'rd', 'drive', 'dr',
    'lane', 'ln', 'court', 'ct', 'circle', 'cir', 'highway', 'hwy', 'parkway', 'pkwy'
}

FRENCH_REGIONS_CITIES = {
    'nouvelle-aquitaine', 'aquitaine', 'pays de la loire', 'loire', 'hauts-de-france',
    'bordeaux', 'nantes', 'lille', 'dunkerque', 'calais', 'tourcoing', 'saint-nazaire',
    'roubaix', 'merignac', 'pessac', 'talence', 'villeneuve-d\'ascq', 'saint-herblain',
    'reze', 'loire-atlantique', 'gironde', 'nord', 'pas-de-calais'
}

STOP_WORDS = {'bis', 'ter', 'de', 'du', 'des', 'la', 'le', 'l', 'd', 'les', 'au', 'aux', 'et', 'france', 'n'}


def parse_num_and_street(addr: Optional[str]) -> Tuple[Optional[str], Set[str]]:
    """Extracts building number and core street tokens."""
    if not addr:
        return None, set()
    s = unicodedata.normalize('NFKD', str(addr)).encode('ASCII', 'ignore').decode('ASCII').lower()
    toks = re.findall(r'[a-z0-9]+', s)
    num = None
    st_toks = []
    for t in toks:
        if t.isdigit() and len(t) <= 5 and num is None:
            num = str(int(t))
        elif not t.isdigit() and t not in STREET_TYPES and t not in FRENCH_REGIONS_CITIES and t not in STOP_WORDS:
            st_toks.append(t)
    return num, set(st_toks)


def resolve_candidate_exclusivity(
    df_predictions: pd.DataFrame,
    s1_lookup: Dict[str, Dict[str, str]],
    cand_lookup: Dict[str, Dict[str, str]],
    max_matches_per_s1: int = 8,
) -> pd.DataFrame:
    """
    Resolves multi-assigned candidate records using address-and-name affinity.
    df_predictions must contain ['source1_entity_id', 'candidate_entity_id'].
    Returns refined DataFrame with exclusive candidate assignments.
    """
    cand_counts = df_predictions["candidate_entity_id"].value_counts()
    single_cands = set(cand_counts[cand_counts == 1].index)
    
    df_single = df_predictions[df_predictions["candidate_entity_id"].isin(single_cands)].copy()
    df_multi = df_predictions[~df_predictions["candidate_entity_id"].isin(single_cands)].copy()
    
    if len(df_multi) == 0:
        return df_single
        
    s1_ids = df_multi["source1_entity_id"].values
    c_ids = df_multi["candidate_entity_id"].values
    
    scores = []
    for i in range(len(df_multi)):
        s1 = s1_lookup.get(s1_ids[i], {})
        cand = cand_lookup.get(c_ids[i], {})
        
        s1_n = str(s1.get("business_name", "")).lower()
        cand_n = str(cand.get("business_name", "")).lower()
        name_sim = Levenshtein.normalized_similarity(s1_n, cand_n)
        
        s1_num, s1_st = parse_num_and_street(s1.get("business_address", ""))
        c_num, c_st = parse_num_and_street(cand.get("business_address", ""))
        
        addr_score = 0.0
        if s1_num and c_num and s1_num == c_num:
            addr_score += 1.0
        elif s1_num and c_num and s1_num != c_num:
            addr_score -= 1.0
            
        st_inter = len(s1_st & c_st)
        if st_inter > 0:
            addr_score += 0.5 * st_inter
        elif s1_st and c_st and st_inter == 0:
            addr_score -= 0.5
            
        scores.append(name_sim + addr_score)
        
    df_multi["affinity"] = scores
    df_multi_winner = df_multi.sort_values(by=["candidate_entity_id", "affinity"], ascending=[True, False]).drop_duplicates(
        subset=["candidate_entity_id"], keep="first"
    )
    df_multi_valid = df_multi_winner[df_multi_winner["affinity"] >= 0.5][["source1_entity_id", "candidate_entity_id"]]
    
    df_final = pd.concat([df_single[["source1_entity_id", "candidate_entity_id"]], df_multi_valid], ignore_index=True)
    
    if max_matches_per_s1:
        df_final["rn"] = df_final.groupby("source1_entity_id").cumcount() + 1
        df_final = df_final[df_final["rn"] <= max_matches_per_s1][["source1_entity_id", "candidate_entity_id"]]
        
    return df_final
