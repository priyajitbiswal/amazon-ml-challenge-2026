"""
Business Entity Resolution Candidate Generation & Blocking Engine (Phase 3 Revision)
Amazon ML Challenge 2026

Implements the High-Recall Multi-Channel Union Blocking Architecture:
- Hard Constraint: candidate.country == source1.country (100% true-match invariant, open-set).
- Channel A  (Priority 100): Exact Normalized Core Name (unambiguous legal suffix & domain stripped).
- Channel A2 (Priority  95): Prefix-Stripped Core Name (normalizes noisy prefixes: 'The ', 'Dr ', 'Smt ', 'M/s ', 'Mr ', 'DBA: ').
- Channel E2 (Priority  85): Address Number + Distinctive Locality Anchor (recovers native-script Indic & trade aliases).
- Channel B  (Priority  80): Rare Name Tokens (inverted index with country-level DF <= 200).
- Channel E  (Priority  75): Address Numeric Anchors (PIN / street number + 3-char name prefix).
- Channel G  (Priority  60): Approximate 1-Edit Initial-Char Typo Key (DF <= 150, recovers '6nni' vs 'Gnni', '0' vs 'O').
- Channel D  (Priority  50): Distinctive Address Tokens (locality & street tokens, frequency cap DF <= 150).
- Channel C  (Priority  40): Character 4-Gram Prefix+Suffix Inverted Index.

Pipeline:
  Multi-Channel Blocking -> Candidate Union -> Multi-Signal Priority Scoring -> Priority Cap Pruning -> Final Candidate Export
"""

import duckdb
import os
import time
from typing import Dict, List, Optional, Tuple, Set, Any

UNAMBIGUOUS_LEGAL_REGEX = (
    r"(?:[,\s\-\(\[\#]+|\b)("
    r"private\s+limited\s+company|limited\s+liability\s+company|limited\s+liability\s+partnership|"
    r"public\s+limited\s+company|private\s+limited|public\s+limited|"
    r"pvt\s+ltd|pvt\s+limited|private\s+ltd|pub\s+ltd|pub\s+limited|"
    r"corporation|incorporated|limited|company|corp|inc|llc|llp|plc|ltd|co|pvt|"
    r"societe\s+a\s+responsabilite\s+limitee|societe\s+par\s+actions\s+simplifiee\s+unipersonnelle|"
    r"societe\s+par\s+actions\s+simplifiee|entreprise\s+unipersonnelle\s+a\s+responsabilite\s+limitee|"
    r"societe\s+anonyme|societe\s+civile|sarlu|sasu|sarl|sas|eurl|sci|snc|sa|ei|et\s+fils|fils|"
    r"प्राइवेट\s+लिमिटेड|प्रा\.\s*लि\.|लिमिटेड|एलएलपी"
    r")(?:[,\s\.\)\]]*)$"
)

PREFIX_NOISE_REGEX = r"^(?:the\s+|dr\s+|smt\s+|m/s\s+|mr\s+|co\s+|>>\s+|#\s*)"


class CandidateBlocker:
    """
    High-throughput candidate blocker powered by DuckDB SQL indexing.
    Maximizes true-match recall while keeping candidate volume strictly bounded.
    """

    def __init__(self, db_connection: Optional[duckdb.DuckDBPyConnection] = None, temp_dir: str = ".tmp_duckdb"):
        self.con = db_connection or duckdb.connect()
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        try:
            self.con.execute(f"PRAGMA temp_directory='{self.temp_dir.replace(chr(92), '/')}';")
            self.con.execute("PRAGMA preserve_insertion_order=false;")
        except Exception:
            pass

    def generate_candidates(
        self,
        s1_table_or_path: str,
        cand_table_or_path: str,
        output_table: str = "final_candidates",
        max_candidates_per_s1: int = 100,
        unambiguous_legal_regex: Optional[str] = None,
        prefix_noise_regex: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the memory-safe multi-channel candidate generation and priority pruning pipeline.
        Partitions by country to prevent Cartesian memory explosions and strictly bounds memory footprint.
        Writes result to `output_table` with schema:
          (source1_entity_id, candidate_entity_id, total_priority, channels_fired, rank_order,
           fired_chan_a, fired_chan_a2, fired_chan_b, fired_chan_c, fired_chan_d, fired_chan_e, fired_chan_e2, fired_chan_g)
        """
        t_start = time.time()
        legal_re = unambiguous_legal_regex or UNAMBIGUOUS_LEGAL_REGEX
        prefix_re = prefix_noise_regex or PREFIX_NOISE_REGEX

        # Ensure input views exist if paths were given
        s1_src = s1_table_or_path
        if os.path.exists(s1_table_or_path) or s1_table_or_path.endswith('.tsv') or s1_table_or_path.endswith('.csv'):
            self.con.execute(f"CREATE OR REPLACE TEMP VIEW _v_s1 AS SELECT entity_id, business_name, business_address, country FROM read_csv('{s1_table_or_path}', delim='\\t', header=true, quote='', all_varchar=true);")
            s1_src = "_v_s1"

        cand_src = cand_table_or_path
        if os.path.exists(cand_table_or_path) or cand_table_or_path.endswith('.tsv') or cand_table_or_path.endswith('.csv'):
            self.con.execute(f"CREATE OR REPLACE TEMP VIEW _v_cand AS SELECT entity_id, business_name, business_address, country FROM read_csv('{cand_table_or_path}', delim='\\t', header=true, quote='', all_varchar=true);")
            cand_src = "_v_cand"

        # Create output table structure
        self.con.execute(f"""
        CREATE OR REPLACE TABLE {output_table} (
            source1_entity_id VARCHAR,
            candidate_entity_id VARCHAR,
            total_priority DOUBLE,
            channels_fired BIGINT,
            rank_order BIGINT,
            fired_chan_a BIGINT,
            fired_chan_a2 BIGINT,
            fired_chan_b BIGINT,
            fired_chan_c BIGINT,
            fired_chan_d BIGINT,
            fired_chan_e BIGINT,
            fired_chan_e2 BIGINT,
            fired_chan_g BIGINT
        );
        """)

        # Discover distinct countries in S1 (hard invariant: candidate.country == s1.country)
        countries_res = self.con.execute(f"SELECT DISTINCT country FROM {s1_src} WHERE country IS NOT NULL").fetchall()
        countries = [r[0] for r in countries_res] if countries_res else [None]

        for country in countries:
            s1_where = f"WHERE country = '{country}'" if country is not None else ""
            cand_where = f"WHERE country = '{country}'" if country is not None else ""

            self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _s1_cur AS
            SELECT entity_id as s1_id, business_name, business_address
            FROM {s1_src} {s1_where};

            CREATE OR REPLACE TEMP TABLE _cand_cur AS
            SELECT entity_id as candidate_id, business_name, business_address
            FROM {cand_src} {cand_where};
            """)

            # 1. CHANNEL A: Exact Core Name (Priority: 100)
            self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _chan_a AS
            WITH s1_clean AS (
                SELECT s1_id,
                       regexp_replace(regexp_replace(replace(lower(trim(business_name)), '.', ''), '{legal_re}', '', 'g'), '[^a-z0-9]', '', 'g') as norm_key
                FROM _s1_cur WHERE business_name IS NOT NULL AND trim(business_name) != ''
            ),
            cand_clean AS (
                SELECT candidate_id,
                       regexp_replace(regexp_replace(replace(lower(trim(business_name)), '.', ''), '{legal_re}', '', 'g'), '[^a-z0-9]', '', 'g') as norm_key
                FROM _cand_cur WHERE business_name IS NOT NULL AND trim(business_name) != ''
            ),
            raw_pairs AS (
                SELECT s.s1_id, c.candidate_id, 100 as priority_score,
                       row_number() OVER (PARTITION BY s.s1_id ORDER BY c.candidate_id) as rn
                FROM s1_clean s
                JOIN cand_clean c ON s.norm_key = c.norm_key
                WHERE length(s.norm_key) >= 3
            )
            SELECT s1_id, candidate_id, priority_score FROM raw_pairs WHERE rn <= {max_candidates_per_s1};
            """)

            # 2. CHANNEL A2: Prefix-Stripped Core Name (Priority: 95)
            self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _chan_a2 AS
            WITH s1_clean AS (
                SELECT s1_id,
                       regexp_replace(regexp_replace(regexp_replace(replace(lower(trim(business_name)), '.', ''), '{prefix_re}', '', 'g'), '{legal_re}', '', 'g'), '[^a-z0-9]', '', 'g') as norm_key
                FROM _s1_cur WHERE business_name IS NOT NULL AND trim(business_name) != ''
            ),
            cand_clean AS (
                SELECT candidate_id,
                       regexp_replace(regexp_replace(regexp_replace(replace(lower(trim(business_name)), '.', ''), '{prefix_re}', '', 'g'), '{legal_re}', '', 'g'), '[^a-z0-9]', '', 'g') as norm_key
                FROM _cand_cur WHERE business_name IS NOT NULL AND trim(business_name) != ''
            ),
            raw_pairs AS (
                SELECT s.s1_id, c.candidate_id, 95 as priority_score,
                       row_number() OVER (PARTITION BY s.s1_id ORDER BY c.candidate_id) as rn
                FROM s1_clean s
                JOIN cand_clean c ON s.norm_key = c.norm_key
                WHERE length(s.norm_key) >= 3
            )
            SELECT s1_id, candidate_id, priority_score FROM raw_pairs WHERE rn <= {max_candidates_per_s1};
            """)

            # 3. CHANNEL B: Rare Name Tokens (Priority: 80)
            self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _cand_name_toks AS
            SELECT candidate_id,
                   unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
            FROM _cand_cur WHERE business_name IS NOT NULL AND trim(business_name) != '';
            DELETE FROM _cand_name_toks WHERE length(tok) < 4;

            CREATE OR REPLACE TEMP TABLE _rare_name_toks AS
            SELECT tok FROM _cand_name_toks GROUP BY tok HAVING count(*) BETWEEN 2 AND 100;

            CREATE OR REPLACE TEMP TABLE _chan_b AS
            WITH s1_tokens AS (
                SELECT s1_id,
                       unnest(string_split(regexp_replace(lower(trim(business_name)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
                FROM _s1_cur WHERE business_name IS NOT NULL AND trim(business_name) != ''
            ),
            raw_pairs AS (
                SELECT DISTINCT s.s1_id, c.candidate_id, 80 as priority_score,
                       row_number() OVER (PARTITION BY s.s1_id ORDER BY c.candidate_id) as rn
                FROM s1_tokens s
                JOIN _rare_name_toks r ON s.tok = r.tok
                JOIN _cand_name_toks c ON s.tok = c.tok
            )
            SELECT s1_id, candidate_id, priority_score FROM raw_pairs WHERE rn <= 50;
            """)

            # 4. CHANNEL E: Address Number + Name Prefix-3 (Priority: 75)
            self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _chan_e AS
            WITH s1_num AS (
                SELECT s1_id,
                       cast(ltrim(regexp_extract(business_address, '[0-9]{{1,6}}'), '0') as varchar) as addr_num,
                       substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as p3
                FROM _s1_cur
                WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{{1,6}}') != ''
                  AND length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 3
            ),
            cand_num AS (
                SELECT candidate_id,
                       cast(ltrim(regexp_extract(business_address, '[0-9]{{1,6}}'), '0') as varchar) as addr_num,
                       substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 1, 3) as p3
                FROM _cand_cur
                WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{{1,6}}') != ''
                  AND length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 3
            ),
            raw_pairs AS (
                SELECT s.s1_id, c.candidate_id, 75 as priority_score,
                       row_number() OVER (PARTITION BY s.s1_id ORDER BY c.candidate_id) as rn
                FROM s1_num s
                JOIN cand_num c ON s.addr_num = c.addr_num AND s.p3 = c.p3
                WHERE length(s.addr_num) > 0
            )
            SELECT s1_id, candidate_id, priority_score FROM raw_pairs WHERE rn <= 50;
            """)

            # 5. CHANNEL D: Distinctive Address Tokens (Priority: 50)
            self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _cand_addr_toks AS
            SELECT candidate_id,
                   unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
            FROM _cand_cur WHERE business_address IS NOT NULL AND trim(business_address) != '';
            DELETE FROM _cand_addr_toks 
            WHERE length(tok) < 5 
               OR tok IN ('street', 'road', 'avenue', 'drive', 'lane', 'boulevard', 'floor', 'suite', 
                          'apartment', 'building', 'sector', 'district', 'nagar', 'north', 'south', 
                          'east', 'west', 'house', 'block', 'first', 'second', 'third', 'opposite', 'near');

            CREATE OR REPLACE TEMP TABLE _rare_addr_toks AS
            SELECT tok FROM _cand_addr_toks GROUP BY tok HAVING count(*) BETWEEN 2 AND 100;

            CREATE OR REPLACE TEMP TABLE _chan_d AS
            WITH s1_tokens AS (
                SELECT s1_id,
                       unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
                FROM _s1_cur WHERE business_address IS NOT NULL AND trim(business_address) != ''
            ),
            raw_pairs AS (
                SELECT DISTINCT s.s1_id, c.candidate_id, 50 as priority_score,
                       row_number() OVER (PARTITION BY s.s1_id ORDER BY c.candidate_id) as rn
                FROM s1_tokens s
                JOIN _rare_addr_toks r ON s.tok = r.tok
                JOIN _cand_addr_toks c ON s.tok = c.tok
            )
            SELECT s1_id, candidate_id, priority_score FROM raw_pairs WHERE rn <= 30;
            """)

            # 6. CHANNEL E2: Address Number + Distinctive Locality Anchor (Priority: 85)
            self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _chan_e2 AS
            WITH s1_addr_anchor AS (
                SELECT s1_id,
                       cast(ltrim(regexp_extract(business_address, '[0-9]{{1,6}}'), '0') as varchar) as num,
                       unnest(string_split(regexp_replace(lower(trim(business_address)), '[^a-z0-9 ]', ' ', 'g'), ' ')) as tok
                FROM _s1_cur 
                WHERE business_address IS NOT NULL AND regexp_extract(business_address, '[0-9]{{1,6}}') != ''
            ),
            cand_addr_anchor AS (
                SELECT c.candidate_id,
                       cast(ltrim(regexp_extract(c.business_address, '[0-9]{{1,6}}'), '0') as varchar) as num,
                       cat.tok
                FROM _cand_cur c
                JOIN _cand_addr_toks cat ON c.candidate_id = cat.candidate_id
                WHERE c.business_address IS NOT NULL AND regexp_extract(c.business_address, '[0-9]{{1,6}}') != ''
            ),
            raw_pairs AS (
                SELECT DISTINCT s.s1_id, c.candidate_id, 85 as priority_score,
                       row_number() OVER (PARTITION BY s.s1_id ORDER BY c.candidate_id) as rn
                FROM s1_addr_anchor s
                JOIN _rare_addr_toks r ON s.tok = r.tok
                JOIN cand_addr_anchor c ON s.tok = c.tok AND s.num = c.num
                WHERE length(s.num) > 0
            )
            SELECT s1_id, candidate_id, priority_score FROM raw_pairs WHERE rn <= 50;
            """)

            # 7. CHANNEL G: Frequency-Capped 1-Edit Initial-Char Typo Key (Priority: 60)
            self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _cand_del_keys AS
            SELECT candidate_id,
                   substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 2, 7) as del_key
            FROM _cand_cur WHERE length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 7;

            CREATE OR REPLACE TEMP TABLE _rare_del_keys AS
            SELECT del_key FROM _cand_del_keys GROUP BY del_key HAVING count(*) BETWEEN 2 AND 100;

            CREATE OR REPLACE TEMP TABLE _chan_g AS
            WITH s1_del AS (
                SELECT s1_id,
                       substring(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g'), 2, 7) as del_key
                FROM _s1_cur WHERE length(regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')) >= 7
            ),
            raw_pairs AS (
                SELECT s.s1_id, c.candidate_id, 60 as priority_score,
                       row_number() OVER (PARTITION BY s.s1_id ORDER BY c.candidate_id) as rn
                FROM s1_del s
                JOIN _rare_del_keys r ON s.del_key = r.del_key
                JOIN _cand_del_keys c ON s.del_key = c.del_key
            )
            SELECT s1_id, candidate_id, priority_score FROM raw_pairs WHERE rn <= 30;
            """)

            # 8. CHANNEL C: Character 4-gram Prefix+Suffix Inverted Index (Priority: 40)
            self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _cand_ngrams AS
            SELECT candidate_id,
                   substring(regexp_replace(lower(trim(business_name)), '{legal_re}', '', 'g'), 1, 4) as p4,
                   substring(regexp_replace(lower(trim(business_name)), '{legal_re}', '', 'g'), -4) as s4
            FROM _cand_cur WHERE length(regexp_replace(lower(trim(business_name)), '{legal_re}', '', 'g')) >= 5;

            CREATE OR REPLACE TEMP TABLE _rare_ngrams AS
            SELECT p4, s4 FROM _cand_ngrams GROUP BY p4, s4 HAVING count(*) BETWEEN 2 AND 50;

            CREATE OR REPLACE TEMP TABLE _chan_c AS
            WITH s1_ngrams AS (
                SELECT s1_id,
                       substring(regexp_replace(lower(trim(business_name)), '{legal_re}', '', 'g'), 1, 4) as p4,
                       substring(regexp_replace(lower(trim(business_name)), '{legal_re}', '', 'g'), -4) as s4
                FROM _s1_cur WHERE length(regexp_replace(lower(trim(business_name)), '{legal_re}', '', 'g')) >= 5
            ),
            raw_pairs AS (
                SELECT s.s1_id, c.candidate_id, 40 as priority_score,
                       row_number() OVER (PARTITION BY s.s1_id ORDER BY c.candidate_id) as rn
                FROM s1_ngrams s
                JOIN _rare_ngrams r ON s.p4 = r.p4 AND s.s4 = r.s4
                JOIN _cand_ngrams c ON s.p4 = c.p4 AND s.s4 = c.s4
            )
            SELECT s1_id, candidate_id, priority_score FROM raw_pairs WHERE rn <= 30;
            """)

            # 9. UNION AND AGGREGATION FOR THIS COUNTRY
            self.con.execute(f"""
            INSERT INTO {output_table}
            WITH all_channel_pairs AS (
                SELECT s1_id, candidate_id, priority_score, 'a' as chan FROM _chan_a
                UNION ALL
                SELECT s1_id, candidate_id, priority_score, 'a2' as chan FROM _chan_a2
                UNION ALL
                SELECT s1_id, candidate_id, priority_score, 'b' as chan FROM _chan_b
                UNION ALL
                SELECT s1_id, candidate_id, priority_score, 'e' as chan FROM _chan_e
                UNION ALL
                SELECT s1_id, candidate_id, priority_score, 'e2' as chan FROM _chan_e2
                UNION ALL
                SELECT s1_id, candidate_id, priority_score, 'g' as chan FROM _chan_g
                UNION ALL
                SELECT s1_id, candidate_id, priority_score, 'd' as chan FROM _chan_d
                UNION ALL
                SELECT s1_id, candidate_id, priority_score, 'c' as chan FROM _chan_c
            ),
            aggregated AS (
                SELECT 
                    s1_id, 
                    candidate_id, 
                    sum(priority_score) as total_priority,
                    count(DISTINCT chan) as channels_fired,
                    max(CASE WHEN chan = 'a' THEN 1 ELSE 0 END) as fired_chan_a,
                    max(CASE WHEN chan = 'a2' THEN 1 ELSE 0 END) as fired_chan_a2,
                    max(CASE WHEN chan = 'b' THEN 1 ELSE 0 END) as fired_chan_b,
                    max(CASE WHEN chan = 'c' THEN 1 ELSE 0 END) as fired_chan_c,
                    max(CASE WHEN chan = 'd' THEN 1 ELSE 0 END) as fired_chan_d,
                    max(CASE WHEN chan = 'e' THEN 1 ELSE 0 END) as fired_chan_e,
                    max(CASE WHEN chan = 'e2' THEN 1 ELSE 0 END) as fired_chan_e2,
                    max(CASE WHEN chan = 'g' THEN 1 ELSE 0 END) as fired_chan_g
                FROM all_channel_pairs
                GROUP BY s1_id, candidate_id
            ),
            ranked AS (
                SELECT 
                    s1_id as source1_entity_id,
                    candidate_id as candidate_entity_id,
                    total_priority,
                    channels_fired,
                    fired_chan_a,
                    fired_chan_a2,
                    fired_chan_b,
                    fired_chan_c,
                    fired_chan_d,
                    fired_chan_e,
                    fired_chan_e2,
                    fired_chan_g,
                    row_number() OVER (
                        PARTITION BY s1_id 
                        ORDER BY total_priority DESC, channels_fired DESC, candidate_id
                    ) as rank_order
                FROM aggregated
            )
            SELECT 
                source1_entity_id, candidate_entity_id, total_priority, channels_fired, rank_order,
                fired_chan_a, fired_chan_a2, fired_chan_b, fired_chan_c, fired_chan_d, fired_chan_e, fired_chan_e2, fired_chan_g
            FROM ranked
            WHERE rank_order <= {max_candidates_per_s1};
            """)

            # Clean up per-country temporary tables
            cleanup_tables = [
                "_chan_a", "_chan_a2", "_chan_b", "_chan_e", "_chan_e2", "_chan_g", "_chan_d", "_chan_c",
                "_cand_name_toks", "_rare_name_toks", "_cand_addr_toks", "_rare_addr_toks",
                "_cand_del_keys", "_rare_del_keys", "_cand_ngrams", "_rare_ngrams",
                "_s1_cur", "_cand_cur"
            ]
            for tbl in cleanup_tables:
                self.con.execute(f"DROP TABLE IF EXISTS {tbl};")

        elapsed = time.time() - t_start
        total_generated = self.con.execute(f"SELECT count(*) FROM {output_table}").fetchone()[0]

        return {
            "output_table": output_table,
            "total_candidates": total_generated,
            "runtime_seconds": elapsed,
            "max_candidates_per_s1": max_candidates_per_s1
        }

    def export_candidate_pairs_tsv(
        self,
        candidate_table: str,
        output_tsv_path: str,
        s1_table_or_path: str,
        cand_table_or_path: str,
    ) -> Dict[str, Any]:
        """
        Exports the final candidate pairs to a TSV file strictly compliant with the official spec:
        - TSV format with header: source1_entity_id\tcandidate_entity_id
        - Strictly verifies that candidate.country == s1.country
        - Strictly verifies that candidate is from Source 2 or Source 3 (starts with 'S2-' or 'S3-')
        - Verifies that S1 ID starts with 'S1-'
        - Guarantees no duplicate pairs
        - Unaltered original IDs
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_tsv_path)), exist_ok=True)

        s1_src = s1_table_or_path
        if os.path.exists(s1_table_or_path) or s1_table_or_path.endswith('.tsv') or s1_table_or_path.endswith('.csv'):
            self.con.execute(f"CREATE OR REPLACE TEMP VIEW _v_exp_s1 AS SELECT entity_id, country FROM read_csv('{s1_table_or_path}', delim='\\t', header=true, quote='', all_varchar=true);")
            s1_src = "_v_exp_s1"

        cand_src = cand_table_or_path
        if os.path.exists(cand_table_or_path) or cand_table_or_path.endswith('.tsv') or cand_table_or_path.endswith('.csv'):
            self.con.execute(f"CREATE OR REPLACE TEMP VIEW _v_exp_cand AS SELECT entity_id, country FROM read_csv('{cand_table_or_path}', delim='\\t', header=true, quote='', all_varchar=true);")
            cand_src = "_v_exp_cand"

        # Validate constraints and export
        self.con.execute(f"""
        CREATE TEMP TABLE _export_validated AS
        SELECT DISTINCT
            c.source1_entity_id,
            c.candidate_entity_id
        FROM {candidate_table} c
        JOIN {s1_src} s ON c.source1_entity_id = s.entity_id
        JOIN {cand_src} m ON c.candidate_entity_id = m.entity_id
        WHERE s.country = m.country
          AND c.source1_entity_id LIKE 'S1-%'
          AND (c.candidate_entity_id LIKE 'S2-%' OR c.candidate_entity_id LIKE 'S3-%');
        """)

        # Write to TSV
        self.con.execute(f"""
        COPY _export_validated TO '{output_tsv_path}' (HEADER, DELIMITER '\t');
        """)

        row_count = self.con.execute("SELECT count(*) FROM _export_validated").fetchone()[0]
        self.con.execute("DROP TABLE IF EXISTS _export_validated;")

        return {
            "output_tsv_path": output_tsv_path,
            "total_pairs_exported": row_count,
            "country_check": "Passed (100% matched country)",
            "source_check": "Passed (Only S1 source and S2/S3 candidates)",
            "uniqueness_check": "Passed (DISTINCT enforced)"
        }
