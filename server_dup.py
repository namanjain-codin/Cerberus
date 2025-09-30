#!/usr/bin/env python3
"""
Integrated Behavioral Biometric Authentication Server
Serves both the API backend and HTML frontend
"""

from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
import sqlite3
import json
import hashlib
import numpy as np
from datetime import datetime
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from scipy import stats as scipy_stats
from scipy.stats import entropy as shannon_entropy
import logging
from gmm_keystroke import train_gmm_model
from keystroke_knn import run_keystroke_knn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiometricAuthenticator:
    def __init__(self, db_path='biometric_auth.db'):
        self.db_path = db_path
        self.init_database()
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.load_models()

    def init_database(self):
        """Initialize the database with required tables"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=20.0)
            # Use WAL mode for better concurrency and prevent locking
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            
            cursor = conn.cursor()

            # Create users table (device fingerprint removed)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create behavioral patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS behavioral_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    pattern_type TEXT NOT NULL,
                    features TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)

            # Create authentication logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS auth_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    username TEXT,
                    success BOOLEAN,
                    confidence_score REAL,
                    patterns_matched INTEGER,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)

            conn.commit()
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

    def extract_features(self, patterns):
        """Extract numerical features from behavioral patterns"""
        features = []
        feature_names = []

        # Device fingerprinting removed

        # Keystroke features
        if 'keystroke' in patterns and patterns['keystroke']:
            keystroke = patterns['keystroke']

            # Advanced keystroke metrics (robust typing-centric features)
            try:
                adv = self._compute_advanced_keystroke_features(keystroke)
            except Exception:
                adv = {}
            for name in [
                'ppd_avg','rpd_avg','rrd_avg',
                'pause_count','avg_burst_length',
                'dwell_skewness','dwell_kurtosis','flight_skewness','flight_kurtosis',
                'flight_time_entropy','htp_avg','ftp_avg',
                'backspace_rate','delete_rate','arrow_key_rate',
                'flight_space_avg','flight_punct_avg',
                'dg_th_avg','dg_he_avg','dg_in_avg','dg_er_avg','dg_an_avg','dg_re_avg','dg_on_avg','dg_at_avg','dg_en_avg','dg_nd_avg',
                'tg_the_pp_avg','tg_and_pp_avg','tg_ing_pp_avg','tg_ent_pp_avg','tg_ion_pp_avg'
            ]:
                features.append(adv.get(name, 0))
                feature_names.append(name)

            # Average dwell times for common keys
            common_keys = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
            for key in common_keys:
                avg_dwell = keystroke.get('avgDwellTimes', {}).get(key, 100)
                features.append(avg_dwell)
                feature_names.append(f'dwell_time_{key}')

            # Typing speed
            features.append(keystroke.get('typingSpeed', 0))
            feature_names.append('typing_speed')

            # Flight times for common key pairs
            common_pairs = ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'ba', 'bb', 'bc', 'bd', 'be', 'bf', 'bg', 'bh', 'bi', 'bj', 'bk', 'bl', 'bm', 'bn', 'bo', 'bp', 'bq', 'br', 'bs', 'bt', 'bu', 'bv', 'bw', 'bx', 'by', 'bz', 'ca', 'cb', 'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'ci', 'cj', 'ck', 'cl', 'cm', 'cn', 'co', 'cp', 'cq', 'cr', 'cs', 'ct', 'cu', 'cv', 'cw', 'cx', 'cy', 'cz', 'da', 'db', 'dc', 'dd', 'de', 'df', 'dg', 'dh', 'di', 'dj', 'dk', 'dl', 'dm', 'dn', 'do', 'dp', 'dq', 'dr', 'ds', 'dt', 'du', 'dv', 'dw', 'dx', 'dy', 'dz', 'ea', 'eb', 'ec', 'ed', 'ee', 'ef', 'eg', 'eh', 'ei', 'ej', 'ek', 'el', 'em', 'en', 'eo', 'ep', 'eq', 'er', 'es', 'et', 'eu', 'ev', 'ew', 'ex', 'ey', 'ez', 'fa', 'fb', 'fc', 'fd', 'fe', 'ff', 'fg', 'fh', 'fi', 'fj', 'fk', 'fl', 'fm', 'fn', 'fo', 'fp', 'fq', 'fr', 'fs', 'ft', 'fu', 'fv', 'fw', 'fx', 'fy', 'fz', 'ga', 'gb', 'gc', 'gd', 'ge', 'gf', 'gg', 'gh', 'gi', 'gj', 'gk', 'gl', 'gm', 'gn', 'go', 'gp', 'gq', 'gr', 'gs', 'gt', 'gu', 'gv', 'gw', 'gx', 'gy', 'gz', 'ha', 'hb', 'hc', 'hd', 'he', 'hf', 'hg', 'hh', 'hi', 'hj', 'hk', 'hl', 'hm', 'hn', 'ho', 'hp', 'hq', 'hr', 'hs', 'ht', 'hu', 'hv', 'hw', 'hx', 'hy', 'hz', 'ia', 'ib', 'ic', 'id', 'ie', 'if', 'ig', 'ih', 'ii', 'ij', 'ik', 'il', 'im', 'in', 'io', 'ip', 'iq', 'ir', 'is', 'it', 'iu', 'iv', 'iw', 'ix', 'iy', 'iz', 'ja', 'jb', 'jc', 'jd', 'je', 'jf', 'jg', 'jh', 'ji', 'jj', 'jk', 'jl', 'jm', 'jn', 'jo', 'jp', 'jq', 'jr', 'js', 'jt', 'ju', 'jv', 'jw', 'jx', 'jy', 'jz', 'ka', 'kb', 'kc', 'kd', 'ke', 'kf', 'kg', 'kh', 'ki', 'kj', 'kk', 'kl', 'km', 'kn', 'ko', 'kp', 'kq', 'kr', 'ks', 'kt', 'ku', 'kv', 'kw', 'kx', 'ky', 'kz', 'la', 'lb', 'lc', 'ld', 'le', 'lf', 'lg', 'lh', 'li', 'lj', 'lk', 'll', 'lm', 'ln', 'lo', 'lp', 'lq', 'lr', 'ls', 'lt', 'lu', 'lv', 'lw', 'lx', 'ly', 'lz', 'ma', 'mb', 'mc', 'md', 'me', 'mf', 'mg', 'mh', 'mi', 'mj', 'mk', 'ml', 'mm', 'mn', 'mo', 'mp', 'mq', 'mr', 'ms', 'mt', 'mu', 'mv', 'mw', 'mx', 'my', 'mz', 'na', 'nb', 'nc', 'nd', 'ne', 'nf', 'ng', 'nh', 'ni', 'nj', 'nk', 'nl', 'nm', 'nn', 'no', 'np', 'nq', 'nr', 'ns', 'nt', 'nu', 'nv', 'nw', 'nx', 'ny', 'nz', 'oa', 'ob', 'oc', 'od', 'oe', 'of', 'og', 'oh', 'oi', 'oj', 'ok', 'ol', 'om', 'on', 'oo', 'op', 'oq', 'or', 'os', 'ot', 'ou', 'ov', 'ow', 'ox', 'oy', 'oz', 'pa', 'pb', 'pc', 'pd', 'pe', 'pf', 'pg', 'ph', 'pi', 'pj', 'pk', 'pl', 'pm', 'pn', 'po', 'pp', 'pq', 'pr', 'ps', 'pt', 'pu', 'pv', 'pw', 'px', 'py', 'pz', 'qa', 'qb', 'qc', 'qd', 'qe', 'qf', 'qg', 'qh', 'qi', 'qj', 'qk', 'ql', 'qm', 'qn', 'qo', 'qp', 'qq', 'qr', 'qs', 'qt', 'qu', 'qv', 'qw', 'qx', 'qy', 'qz', 'ra', 'rb', 'rc', 'rd', 're', 'rf', 'rg', 'rh', 'ri', 'rj', 'rk', 'rl', 'rm', 'rn', 'ro', 'rp', 'rq', 'rr', 'rs', 'rt', 'ru', 'rv', 'rw', 'rx', 'ry', 'rz', 'sa', 'sb', 'sc', 'sd', 'se', 'sf', 'sg', 'sh', 'si', 'sj', 'sk', 'sl', 'sm', 'sn', 'so', 'sp', 'sq', 'sr', 'ss', 'st', 'su', 'sv', 'sw', 'sx', 'sy', 'sz', 'ta', 'tb', 'tc', 'td', 'te', 'tf', 'tg', 'th', 'ti', 'tj', 'tk', 'tl', 'tm', 'tn', 'to', 'tp', 'tq', 'tr', 'ts', 'tt', 'tu', 'tv', 'tw', 'tx', 'ty', 'tz', 'ua', 'ub', 'uc', 'ud', 'ue', 'uf', 'ug', 'uh', 'ui', 'uj', 'uk', 'ul', 'um', 'un', 'uo', 'up', 'uq', 'ur', 'us', 'ut', 'uu', 'uv', 'uw', 'ux', 'uy', 'uz', 'va', 'vb', 'vc', 'vd', 've', 'vf', 'vg', 'vh', 'vi', 'vj', 'vk', 'vl', 'vm', 'vn', 'vo', 'vp', 'vq', 'vr', 'vs', 'vt', 'vu', 'vv', 'vw', 'vx', 'vy', 'vz', 'wa', 'wb', 'wc', 'wd', 'we', 'wf', 'wg', 'wh', 'wi', 'wj', 'wk', 'wl', 'wm', 'wn', 'wo', 'wp', 'wq', 'wr', 'ws', 'wt', 'wu', 'wv', 'ww', 'wx', 'wy', 'wz', 'xa', 'xb', 'xc', 'xd', 'xe', 'xf', 'xg', 'xh', 'xi', 'xj', 'xk', 'xl', 'xm', 'xn', 'xo', 'xp', 'xq', 'xr', 'xs', 'xt', 'xu', 'xv', 'xw', 'xx', 'xy', 'xz', 'ya', 'yb', 'yc', 'yd', 'ye', 'yf', 'yg', 'yh', 'yi', 'yj', 'yk', 'yl', 'ym', 'yn', 'yo', 'yp', 'yq', 'yr', 'ys', 'yt', 'yu', 'yv', 'yw', 'yx', 'yy', 'yz', 'za', 'zb', 'zc', 'zd', 'ze', 'zf', 'zg', 'zh', 'zi', 'zj', 'zk', 'zl', 'zm', 'zn', 'zo', 'zp', 'zq', 'zr', 'zs', 'zt', 'zu', 'zv', 'zw', 'zx', 'zy', 'zz']
            for pair in common_pairs:
                flight_time = keystroke.get('avgFlightTimes', {}).get(f'{pair[0]}_{pair[1]}', 50)
                features.append(flight_time)
                feature_names.append(f'flight_time_{pair}')
        else:
            # Pad with zeros if no keystroke data
            features.extend([0] * 26)  # 10 dwell + 1 speed + 5 flight times
            feature_names.extend([f'dwell_time_{k}' for k in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']])
            feature_names.extend(['typing_speed'])
            feature_names.extend([f'flight_time_{p}' for p in ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'ba', 'bb', 'bc', 'bd', 'be', 'bf', 'bg', 'bh', 'bi', 'bj', 'bk', 'bl', 'bm', 'bn', 'bo', 'bp', 'bq', 'br', 'bs', 'bt', 'bu', 'bv', 'bw', 'bx', 'by', 'bz', 'ca', 'cb', 'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'ci', 'cj', 'ck', 'cl', 'cm', 'cn', 'co', 'cp', 'cq', 'cr', 'cs', 'ct', 'cu', 'cv', 'cw', 'cx', 'cy', 'cz', 'da', 'db', 'dc', 'dd', 'de', 'df', 'dg', 'dh', 'di', 'dj', 'dk', 'dl', 'dm', 'dn', 'do', 'dp', 'dq', 'dr', 'ds', 'dt', 'du', 'dv', 'dw', 'dx', 'dy', 'dz', 'ea', 'eb', 'ec', 'ed', 'ee', 'ef', 'eg', 'eh', 'ei', 'ej', 'ek', 'el', 'em', 'en', 'eo', 'ep', 'eq', 'er', 'es', 'et', 'eu', 'ev', 'ew', 'ex', 'ey', 'ez', 'fa', 'fb', 'fc', 'fd', 'fe', 'ff', 'fg', 'fh', 'fi', 'fj', 'fk', 'fl', 'fm', 'fn', 'fo', 'fp', 'fq', 'fr', 'fs', 'ft', 'fu', 'fv', 'fw', 'fx', 'fy', 'fz', 'ga', 'gb', 'gc', 'gd', 'ge', 'gf', 'gg', 'gh', 'gi', 'gj', 'gk', 'gl', 'gm', 'gn', 'go', 'gp', 'gq', 'gr', 'gs', 'gt', 'gu', 'gv', 'gw', 'gx', 'gy', 'gz', 'ha', 'hb', 'hc', 'hd', 'he', 'hf', 'hg', 'hh', 'hi', 'hj', 'hk', 'hl', 'hm', 'hn', 'ho', 'hp', 'hq', 'hr', 'hs', 'ht', 'hu', 'hv', 'hw', 'hx', 'hy', 'hz', 'ia', 'ib', 'ic', 'id', 'ie', 'if', 'ig', 'ih', 'ii', 'ij', 'ik', 'il', 'im', 'in', 'io', 'ip', 'iq', 'ir', 'is', 'it', 'iu', 'iv', 'iw', 'ix', 'iy', 'iz', 'ja', 'jb', 'jc', 'jd', 'je', 'jf', 'jg', 'jh', 'ji', 'jj', 'jk', 'jl', 'jm', 'jn', 'jo', 'jp', 'jq', 'jr', 'js', 'jt', 'ju', 'jv', 'jw', 'jx', 'jy', 'jz', 'ka', 'kb', 'kc', 'kd', 'ke', 'kf', 'kg', 'kh', 'ki', 'kj', 'kk', 'kl', 'km', 'kn', 'ko', 'kp', 'kq', 'kr', 'ks', 'kt', 'ku', 'kv', 'kw', 'kx', 'ky', 'kz', 'la', 'lb', 'lc', 'ld', 'le', 'lf', 'lg', 'lh', 'li', 'lj', 'lk', 'll', 'lm', 'ln', 'lo', 'lp', 'lq', 'lr', 'ls', 'lt', 'lu', 'lv', 'lw', 'lx', 'ly', 'lz', 'ma', 'mb', 'mc', 'md', 'me', 'mf', 'mg', 'mh', 'mi', 'mj', 'mk', 'ml', 'mm', 'mn', 'mo', 'mp', 'mq', 'mr', 'ms', 'mt', 'mu', 'mv', 'mw', 'mx', 'my', 'mz', 'na', 'nb', 'nc', 'nd', 'ne', 'nf', 'ng', 'nh', 'ni', 'nj', 'nk', 'nl', 'nm', 'nn', 'no', 'np', 'nq', 'nr', 'ns', 'nt', 'nu', 'nv', 'nw', 'nx', 'ny', 'nz', 'oa', 'ob', 'oc', 'od', 'oe', 'of', 'og', 'oh', 'oi', 'oj', 'ok', 'ol', 'om', 'on', 'oo', 'op', 'oq', 'or', 'os', 'ot', 'ou', 'ov', 'ow', 'ox', 'oy', 'oz', 'pa', 'pb', 'pc', 'pd', 'pe', 'pf', 'pg', 'ph', 'pi', 'pj', 'pk', 'pl', 'pm', 'pn', 'po', 'pp', 'pq', 'pr', 'ps', 'pt', 'pu', 'pv', 'pw', 'px', 'py', 'pz', 'qa', 'qb', 'qc', 'qd', 'qe', 'qf', 'qg', 'qh', 'qi', 'qj', 'qk', 'ql', 'qm', 'qn', 'qo', 'qp', 'qq', 'qr', 'qs', 'qt', 'qu', 'qv', 'qw', 'qx', 'qy', 'qz', 'ra', 'rb', 'rc', 'rd', 're', 'rf', 'rg', 'rh', 'ri', 'rj', 'rk', 'rl', 'rm', 'rn', 'ro', 'rp', 'rq', 'rr', 'rs', 'rt', 'ru', 'rv', 'rw', 'rx', 'ry', 'rz', 'sa', 'sb', 'sc', 'sd', 'se', 'sf', 'sg', 'sh', 'si', 'sj', 'sk', 'sl', 'sm', 'sn', 'so', 'sp', 'sq', 'sr', 'ss', 'st', 'su', 'sv', 'sw', 'sx', 'sy', 'sz', 'ta', 'tb', 'tc', 'td', 'te', 'tf', 'tg', 'th', 'ti', 'tj', 'tk', 'tl', 'tm', 'tn', 'to', 'tp', 'tq', 'tr', 'ts', 'tt', 'tu', 'tv', 'tw', 'tx', 'ty', 'tz', 'ua', 'ub', 'uc', 'ud', 'ue', 'uf', 'ug', 'uh', 'ui', 'uj', 'uk', 'ul', 'um', 'un', 'uo', 'up', 'uq', 'ur', 'us', 'ut', 'uu', 'uv', 'uw', 'ux', 'uy', 'uz', 'va', 'vb', 'vc', 'vd', 've', 'vf', 'vg', 'vh', 'vi', 'vj', 'vk', 'vl', 'vm', 'vn', 'vo', 'vp', 'vq', 'vr', 'vs', 'vt', 'vu', 'vv', 'vw', 'vx', 'vy', 'vz', 'wa', 'wb', 'wc', 'wd', 'we', 'wf', 'wg', 'wh', 'wi', 'wj', 'wk', 'wl', 'wm', 'wn', 'wo', 'wp', 'wq', 'wr', 'ws', 'wt', 'wu', 'wv', 'ww', 'wx', 'wy', 'wz', 'xa', 'xb', 'xc', 'xd', 'xe', 'xf', 'xg', 'xh', 'xi', 'xj', 'xk', 'xl', 'xm', 'xn', 'xo', 'xp', 'xq', 'xr', 'xs', 'xt', 'xu', 'xv', 'xw', 'xx', 'xy', 'xz', 'ya', 'yb', 'yc', 'yd', 'ye', 'yf', 'yg', 'yh', 'yi', 'yj', 'yk', 'yl', 'ym', 'yn', 'yo', 'yp', 'yq', 'yr', 'ys', 'yt', 'yu', 'yv', 'yw', 'yx', 'yy', 'yz', 'za', 'zb', 'zc', 'zd', 'ze', 'zf', 'zg', 'zh', 'zi', 'zj', 'zk', 'zl', 'zm', 'zn', 'zo', 'zp', 'zq', 'zr', 'zs', 'zt', 'zu', 'zv', 'zw', 'zx', 'zy', 'zz']])

        # Mouse features
        if 'mouse' in patterns and patterns['mouse']:
            mouse = patterns['mouse']
            features.extend([
                mouse.get('avgMouseSpeed', 0),
                mouse.get('avgMouseAcceleration', 0),
                mouse.get('totalMouseDistance', 0),
                mouse.get('mouseMovements', 0),
                mouse.get('clickCount', 0)
            ])
            feature_names.extend([
                'avg_mouse_speed', 'avg_mouse_acceleration', 
                'total_mouse_distance', 'mouse_movements', 'click_count'
            ])
        else:
            features.extend([0] * 5)
            feature_names.extend([
                'avg_mouse_speed', 'avg_mouse_acceleration',
                'total_mouse_distance', 'mouse_movements', 'click_count'
            ])

        # Touch/swipe gesture features removed

        # Debug: Log feature extraction
        logger.info(f"Extracted {len(features)} features")
        logger.info(f"Feature names: {feature_names[:10] if len(feature_names) > 10 else feature_names}")
        logger.info(f"Feature values sample: {features[:10] if len(features) > 10 else features}")
        
        return np.array(features), feature_names

    def _compute_advanced_keystroke_features(self, keystroke_obj: dict):
        """Compute PP/RP/RR, pause/burst, error rates, distribution stats, entropy, and proportions.
        Expects keystroke_obj to contain rawEvents (list) and typedString (str) optionally.
        """
        out = {}
        raw = keystroke_obj.get('rawEvents') or []
        typed = keystroke_obj.get('typedString') or ''

        # Build per-key instance timings (first-order, simple implementation)
        press_times = []
        release_times = []
        dwell_times_seq = []
        flight_times_seq = []

        last_release = None
        # Map index to event for sequence handling
        for i, ev in enumerate(raw):
            if ev.get('type') == 'down':
                # find matching up
                up_time = None
                for j in range(i + 1, min(i + 80, len(raw))):  # bounded search
                    ev2 = raw[j]
                    if ev2.get('type') == 'up' and ev2.get('key') == ev.get('key'):
                        up_time = ev2.get('timestamp')
                        break
                if up_time is not None:
                    hold = max(0, up_time - ev.get('timestamp'))
                    dwell_times_seq.append(hold)
                press_times.append({'key': ev.get('key'), 'press': ev.get('timestamp'), 'release': up_time})
                if last_release is not None:
                    flight_times_seq.append(max(0, ev.get('timestamp') - last_release))
            elif ev.get('type') == 'up':
                last_release = ev.get('timestamp')
                release_times.append(ev.get('timestamp'))

        # Compute PP/RP/RR on consecutive chars in typed string when available
        ppd_list, rpd_list, rrd_list = [], [], []
        digraph_map = {}  # key pair -> list of flight (RPD)
        trigraph_map = {} # three keys -> list of PP over k and k+2
        if typed and len(typed) >= 2:
            # Build a simple mapping of first occurrence times (approx)
            key_index = 0
            time_map = {}
            for ev in raw:
                if ev.get('type') == 'down' and key_index < len(typed):
                    k = typed[key_index]
                    time_map.setdefault(key_index, {'key': k, 'press': ev.get('timestamp')})
                if ev.get('type') == 'up' and key_index < len(typed):
                    if key_index in time_map and 'release' not in time_map[key_index]:
                        time_map[key_index]['release'] = ev.get('timestamp')
                        key_index += 1
            for i in range(len(typed) - 1):
                t1 = time_map.get(i)
                t2 = time_map.get(i + 1)
                if t1 and t2 and 'press' in t1 and 'release' in t1 and 'press' in t2 and 'release' in t2:
                    ppd_list.append(t2['press'] - t1['press'])
                    rpd_list.append(t2['press'] - t1['release'])
                    rrd_list.append(t2['release'] - t1['release'])
                    dg = f"{t1['key']}{t2['key']}"
                    digraph_map.setdefault(dg, []).append(t2['press'] - t1['release'])
            # Trigraphs (press i to press i+2)
            for i in range(len(typed) - 2):
                t1 = time_map.get(i)
                t3 = time_map.get(i + 2)
                if t1 and t3 and 'press' in t1 and 'press' in t3:
                    tg = f"{t1['key']}{time_map.get(i+1,{}).get('key','')}{t3['key']}"
                    trigraph_map.setdefault(tg, []).append(t3['press'] - t1['press'])

        out['ppd_avg'] = float(np.mean(ppd_list)) if ppd_list else 0.0
        out['rpd_avg'] = float(np.mean(rpd_list)) if rpd_list else 0.0
        out['rrd_avg'] = float(np.mean(rrd_list)) if rrd_list else 0.0

        # Pause and burst metrics
        pause_threshold = 200.0
        bursts = []
        current_burst = 0
        for ft in flight_times_seq:
            if ft > pause_threshold:
                if current_burst > 0:
                    bursts.append(current_burst)
                current_burst = 0
            else:
                current_burst += 1
        if current_burst > 0:
            bursts.append(current_burst)
        out['pause_count'] = int(len([ft for ft in flight_times_seq if ft > pause_threshold]))
        out['avg_burst_length'] = float(np.mean(bursts)) if bursts else 0.0

        # Distributional stats
        if len(dwell_times_seq) > 1:
            out['dwell_skewness'] = float(scipy_stats.skew(dwell_times_seq))
            out['dwell_kurtosis'] = float(scipy_stats.kurtosis(dwell_times_seq))
        else:
            out['dwell_skewness'] = 0.0
            out['dwell_kurtosis'] = 0.0
        if len(flight_times_seq) > 1:
            out['flight_skewness'] = float(scipy_stats.skew(flight_times_seq))
            out['flight_kurtosis'] = float(scipy_stats.kurtosis(flight_times_seq))
        else:
            out['flight_skewness'] = 0.0
            out['flight_kurtosis'] = 0.0

        # Entropy of flight time distribution
        if len(flight_times_seq) > 1:
            hist, _ = np.histogram(np.array(flight_times_seq), bins=10, density=True)
            # Avoid zero-only hist
            if np.any(hist):
                out['flight_time_entropy'] = float(shannon_entropy(hist, base=2))
            else:
                out['flight_time_entropy'] = 0.0
        else:
            out['flight_time_entropy'] = 0.0

        # Proportion features (HTP/FTP) using successive pairs where possible
        htp_vals, ftp_vals = [] , []
        # align dwell and flight by sequence best-effort
        m = min(len(dwell_times_seq), len(flight_times_seq))
        for i in range(m):
            ht = dwell_times_seq[i]
            ft = flight_times_seq[i]
            cycle = ht + ft
            if cycle > 0:
                htp_vals.append(ht / cycle)
                ftp_vals.append(ft / cycle)
        out['htp_avg'] = float(np.mean(htp_vals)) if htp_vals else 0.0
        out['ftp_avg'] = float(np.mean(ftp_vals)) if ftp_vals else 0.0

        # Error/correction behavior
        error_counts = {'Backspace': 0, 'Delete': 0, 'ArrowLeft': 0, 'ArrowRight': 0}
        char_count = 0
        for ev in raw:
            if ev.get('type') == 'down':
                k = ev.get('key')
                if isinstance(k, str) and len(k) == 1:
                    char_count += 1
                if k in error_counts:
                    error_counts[k] += 1
        total_chars = char_count if char_count > 0 else 1
        out['backspace_rate'] = error_counts['Backspace'] / total_chars
        out['delete_rate'] = error_counts['Delete'] / total_chars
        out['arrow_key_rate'] = (error_counts['ArrowLeft'] + error_counts['ArrowRight']) / total_chars

        # Punctuation/space timing (average flight into/out of space)
        space_flights = []
        punct_flights = []
        for i in range(len(typed) - 1):
            # approximate using digraph_map entries built above
            pair = f"{typed[i]}{typed[i+1]}"
            vals = digraph_map.get(pair, [])
            if not vals:
                continue
            if typed[i+1] == ' ' or typed[i] == ' ':
                space_flights.extend(vals)
            if typed[i] in ',.;:' or typed[i+1] in ',.;:':
                punct_flights.extend(vals)
        out['flight_space_avg'] = float(np.mean(space_flights)) if space_flights else 0.0
        out['flight_punct_avg'] = float(np.mean(punct_flights)) if punct_flights else 0.0

        # Fixed-set digraph/trigraph averages to feed classifier (top English digraphs)
        top_digraphs = ['th','he','in','er','an','re','on','at','en','nd']
        for dg in top_digraphs:
            vals = digraph_map.get(dg, [])
            out[f'dg_{dg}_avg'] = float(np.mean(vals)) if vals else 0.0
        top_trigraphs = ['the','and','ing','ent','ion']
        for tg in top_trigraphs:
            vals = trigraph_map.get(tg, [])
            out[f'tg_{tg}_pp_avg'] = float(np.mean(vals)) if vals else 0.0

        return out

    def register_user(self, username, password, patterns):
        """Register a new user with their behavioral patterns"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=20.0)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()

            # Hash password
            password_hash = hashlib.sha256(password.encode()).hexdigest()

            # Insert user (device fingerprint removed)
            cursor.execute("""
                INSERT INTO users (username, password_hash)
                VALUES (?, ?)
            """, (username, password_hash))

            user_id = cursor.lastrowid

            # Extract and store behavioral patterns
            features, feature_names = self.extract_features(patterns)

            # Store individual pattern types (touch removed)
            for pattern_type in ['keystroke', 'mouse']:
                if pattern_type in patterns:
                    cursor.execute("""
                        INSERT INTO behavioral_patterns (user_id, pattern_type, features)
                        VALUES (?, ?, ?)
                    """, (user_id, pattern_type, json.dumps(patterns[pattern_type])))

            # Store combined features
            cursor.execute("""
                INSERT INTO behavioral_patterns (user_id, pattern_type, features)
                VALUES (?, ?, ?)
            """, (user_id, 'combined_features', json.dumps({
                'features': features.tolist(),
                'feature_names': feature_names
            })))

            conn.commit()

            logger.info(f"User {username} registered successfully")
            return True, "Registration successful"

        except sqlite3.IntegrityError:
            return False, "Username already exists"
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                logger.error(f"Database locked during registration for {username}")
                return False, "Database temporarily unavailable, please try again"
            else:
                logger.error(f"Database error during registration: {str(e)}")
                return False, f"Database error: {str(e)}"
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            return False, f"Registration failed: {str(e)}"
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

    def authenticate_user(self, username, password, patterns):
        """Authenticate user based on password and behavioral patterns"""
        conn = None
        try:
            # Use timeout and better connection handling
            conn = sqlite3.connect(self.db_path, timeout=20.0)
            conn.execute("PRAGMA journal_mode=WAL")  # Use WAL mode for better concurrency
            cursor = conn.cursor()

            # Verify password if provided
            if password:
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                cursor.execute("""
                    SELECT id FROM users 
                    WHERE username = ? AND password_hash = ?
                """, (username, password_hash))
            else:
                # For biometric-only auth, just get user ID
                cursor.execute("""
                    SELECT id FROM users 
                    WHERE username = ?
                """, (username,))

            user_result = cursor.fetchone()
            if not user_result:
                return False, 0.0, "Invalid credentials"

            user_id = user_result[0]

            # Get stored behavioral patterns
            cursor.execute("""
                SELECT features FROM behavioral_patterns 
                WHERE user_id = ? AND pattern_type = 'combined_features'
                ORDER BY created_at DESC
            """, (user_id,))

            stored_patterns = cursor.fetchall()
            if not stored_patterns:
                return False, 0.0, "No behavioral patterns found"

            # Extract features from current patterns
            current_features, current_feature_names = self.extract_features(patterns)

            # Calculate similarity scores with stored patterns
            similarity_scores = []
            for pattern_data in stored_patterns[:5]:  # Use last 5 patterns
                stored_data = json.loads(pattern_data[0])
                stored_features = np.array(stored_data['features'])

                # Ensure same feature count
                min_len = min(len(current_features), len(stored_features))
                current_trimmed = current_features[:min_len]
                stored_trimmed = stored_features[:min_len]
                
                # Debug: Check for NaN or infinite values
                if np.any(np.isnan(current_trimmed)) or np.any(np.isnan(stored_trimmed)):
                    logger.warning(f"NaN values found in features for {username}")
                    similarity = 0.0
                elif np.any(np.isinf(current_trimmed)) or np.any(np.isinf(stored_trimmed)):
                    logger.warning(f"Inf values found in features for {username}")
                    similarity = 0.0
                else:
                    # Use a simple percentage-based similarity approach
                    # Calculate how many features are "close enough"
                    tolerance = 0.1  # 10% tolerance for each feature
                    close_features = 0
                    total_features = len(current_trimmed)
                    
                    for i in range(total_features):
                        current_val = current_trimmed[i]
                        stored_val = stored_trimmed[i]
                        
                        # Calculate relative difference
                        if stored_val != 0:
                            relative_diff = abs(current_val - stored_val) / abs(stored_val)
                        else:
                            relative_diff = abs(current_val) if current_val != 0 else 0
                        
                        if relative_diff <= tolerance:
                            close_features += 1
                    
                    # Similarity is the percentage of features that are close
                    similarity = close_features / total_features if total_features > 0 else 0.0
                    
                    # Debug logging
                    logger.info(f"Close features: {close_features}/{total_features}, Similarity: {similarity:.4f}")
                similarity_scores.append(similarity)

            # Average similarity score
            avg_similarity = np.mean(similarity_scores)
            
            # Log similarity scores for debugging
            logger.info(f"Similarity scores for {username}: {[f'{s:.4f}' for s in similarity_scores]}")
            logger.info(f"Average similarity: {avg_similarity:.4f}")
            logger.info(f"Feature count: {len(current_features)}")
            logger.info(f"Current features sample: {current_features[:10] if len(current_features) > 10 else current_features}")
            logger.info(f"Stored patterns count: {len(stored_patterns)}")

            # Device fingerprint comparison removed

            # Confidence score based solely on behavioral similarity
            confidence_score = avg_similarity
            
            # Remove artificial minimums - let the natural similarity score stand
            # Only apply minimal bounds to prevent invalid values
            
            # Allow very low scores for poor matches - no artificial floor
            # Keep score within [0,1] bounds only
            if confidence_score < 0.0:
                confidence_score = 0.0
            elif confidence_score > 1.0:
                confidence_score = 1.0

            # Enhanced authentication threshold with stricter requirements
            auth_threshold = 0.85  # Increased from 0.65 to 0.85 for better security
            is_authenticated = confidence_score >= auth_threshold
            
            # Additional security checks
            security_flags = []
            
            # Check for suspicious patterns
            if avg_similarity < 0.3:
                security_flags.append("LOW_BEHAVIORAL_SIMILARITY")
            
            # Device mismatch flag removed
            
            # If behavioral auth fails but confidence is above 0.5, flag for enhanced verification
            if not is_authenticated and confidence_score >= 0.5:
                security_flags.append("REQUIRES_ENHANCED_VERIFICATION")
            
            logger.info(f"Confidence calculation: behavioral={avg_similarity:.3f}, final={confidence_score:.3f}")

            # Log authentication attempt
            cursor.execute("""
                INSERT INTO auth_logs (user_id, username, success, confidence_score, 
                                     patterns_matched, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, username, is_authenticated, confidence_score, 
                  len(similarity_scores), 
                  'unknown',  # Would get from request in real app
                  'unknown'))  # Would get from request in real app

            # If authenticated, store new pattern for learning
            if is_authenticated:
                cursor.execute("""
                    INSERT INTO behavioral_patterns (user_id, pattern_type, features)
                    VALUES (?, ?, ?)
                """, (user_id, 'combined_features', json.dumps({
                    'features': current_features.tolist(),
                    'feature_names': current_feature_names
                })))

            conn.commit()

            # Determine authentication result and next steps
            if is_authenticated:
                status_msg = "Authentication successful"
            elif "REQUIRES_ENHANCED_VERIFICATION" in security_flags:
                status_msg = "REQUIRES_ENHANCED_VERIFICATION"
            else:
                status_msg = "Behavioral patterns do not match"
            
            logger.info(f"Authentication attempt for {username}: {is_authenticated} (confidence: {confidence_score:.3f})")

            return is_authenticated, confidence_score, status_msg, security_flags

        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                logger.error(f"Database locked during authentication for {username}")
                return False, 0.0, "Database temporarily unavailable, please try again"
            else:
                logger.error(f"Database error during authentication: {str(e)}")
                return False, 0.0, f"Database error: {str(e)}"
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False, 0.0, f"Authentication failed: {str(e)}"
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

    def save_models(self):
        """Save trained models to disk"""
        try:
            with open('biometric_models.pkl', 'wb') as f:
                pickle.dump({
                    'scaler': self.scaler,
                    'model': self.model,
                    'anomaly_detector': self.anomaly_detector
                }, f)
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")

    def load_models(self):
        """Load trained models from disk"""
        try:
            if os.path.exists('biometric_models.pkl'):
                with open('biometric_models.pkl', 'rb') as f:
                    models = pickle.load(f)
                    self.scaler = models['scaler']
                    self.model = models['model']
                    self.anomaly_detector = models['anomaly_detector']
                logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")

# Create Flask app instance
app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'
CORS(app, supports_credentials=True, resources={r"/api/*": {"origins": ["http://localhost:5000","http://localhost:5001"]}})

# Initialize the authenticator
authenticator = BiometricAuthenticator()

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'biometric_auth_demo.html')

@app.route('/frontend_biometric_capture.js')
def serve_js():
    """Serve the JavaScript file"""
    return send_from_directory('.', 'frontend_biometric_capture.js', mimetype='application/javascript')

@app.route('/favicon.ico')
def serve_favicon():
    """Serve a default favicon or return 204 No Content"""
    return '', 204

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database': 'connected' if os.path.exists(authenticator.db_path) else 'disconnected'
    })

@app.route('/api/register', methods=['POST'])
def register_user():
    """User registration endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({'success': False, 'error': 'Username and password required'}), 400
        
        username = data['username']
        password = data['password']
        patterns = data.get('patterns', {})
        
        # Register the user
        success, message = authenticator.register_user(username, password, patterns)
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'username': username
            })
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 400
            
    except Exception as e:
        logger.error(f"Registration endpoint error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login_user():
    """User login endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({'success': False, 'error': 'Username and password required'}), 400
        
        username = data['username']
        password = data['password']
        patterns = data.get('patterns', {})
        
        # Authenticate the user
        auth_result = authenticator.authenticate_user(username, password, patterns)
        
        if len(auth_result) == 4:
            is_authenticated, confidence_score, message, security_flags = auth_result
        else:
            # Handle old return format for backward compatibility
            is_authenticated, confidence_score, message = auth_result
            security_flags = []
        
        if is_authenticated:
            # Set session
            session['user_id'] = username
            session['authenticated'] = True
            
            return jsonify({
                'success': True,
                'message': message,
                'confidence_score': confidence_score,
                'username': username
            })
        elif message == "REQUIRES_ENHANCED_VERIFICATION":
            # Set session for enhanced verification
            session['user_id'] = username
            session['needs_enhanced_verification'] = True
            
            return jsonify({
                'success': False,
                'requires_enhanced_verification': True,
                'message': 'Enhanced verification required for security',
                'confidence_score': confidence_score,
                'security_flags': security_flags,
                'redirect_url': '/enhanced-auth'
            }), 202  # 202 Accepted - requires additional verification
        else:
            return jsonify({
                'success': False,
                'error': message,
                'confidence_score': confidence_score
            }), 401
            
    except Exception as e:
        logger.error(f"Login endpoint error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/authenticate', methods=['POST'])
def authenticate_biometric():
    """Biometric authentication endpoint"""
    try:
        if not session.get('authenticated'):
            return jsonify({'success': False, 'error': 'No active session'}), 401
        
        data = request.get_json()
        username = session.get('user_id')
        patterns = data.get('patterns', {})
        
        if not patterns:
            return jsonify({'success': False, 'error': 'No behavioral patterns provided'}), 400
        
        # Perform biometric authentication
        is_authenticated, confidence_score, message = authenticator.authenticate_user(
            username, None, patterns
        )
        
        if is_authenticated:
            return jsonify({
                'success': True,
                'message': message,
                'confidence_score': confidence_score,
                'username': username
            })
        else:
            return jsonify({
                'success': False,
                'error': message,
                'confidence_score': confidence_score
            }), 401
            
    except Exception as e:
        logger.error(f"Biometric auth endpoint error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout endpoint"""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        # Get basic stats from the database
        conn = sqlite3.connect(authenticator.db_path)
        cursor = conn.cursor()
        
        # User count
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        # Authentication attempts
        cursor.execute("SELECT COUNT(*) FROM auth_logs")
        total_auths = cursor.fetchone()[0]
        
        # Successful authentications
        cursor.execute("SELECT COUNT(*) FROM auth_logs WHERE success = 1")
        successful_auths = cursor.fetchone()[0]
        
        # Success rate
        success_rate = (successful_auths / total_auths * 100) if total_auths > 0 else 0
        
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': {
                'total_users': user_count,
                'total_authentications': total_auths,
                'successful_authentications': successful_auths,
                'success_rate': round(success_rate, 2)
            }
        })
        
    except Exception as e:
        logger.error(f"Stats endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/gmm-train', methods=['POST'])
def api_gmm_train():
    """Train GMM keystroke model from CSV and return ROC data"""
    try:
        data = request.get_json() or {}
        csv_path = data.get('csv_path')
        M = int(data.get('M', 3))
        delta = float(data.get('delta', 1.0))
        train_ratio = float(data.get('train_ratio', 0.7))
        valid_ratio = float(data.get('valid_ratio', 0.3))

        if not csv_path or not os.path.exists(csv_path):
            return jsonify({'success': False, 'error': 'Valid csv_path required'}), 400

        fpr, tpr, thresholds = train_gmm_model(
            csv_path=csv_path,
            M=M,
            delta=delta,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio
        )

        return jsonify({
            'success': True,
            'fpr': [float(x) for x in fpr.tolist()],
            'tpr': [float(x) for x in tpr.tolist()],
            'thresholds': [float(x) for x in thresholds.tolist()],
            'params': {'M': M, 'delta': delta, 'train_ratio': train_ratio, 'valid_ratio': valid_ratio}
        })

    except Exception as e:
        logger.error(f"GMM train endpoint error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/keystroke-knn', methods=['POST'])
def api_keystroke_knn():
    """Run notebook-inspired KNN keystroke pipeline and return CV accuracy and predictions"""
    try:
        data = request.get_json() or {}
        train_csv = data.get('train_csv')
        test_csv = data.get('test_csv')
        num_bins = int(data.get('num_bins', 10))
        n_neighbors = int(data.get('n_neighbors', 1))

        if not train_csv or not os.path.exists(train_csv):
            return jsonify({'success': False, 'error': 'Valid train_csv required'}), 400
        if not test_csv or not os.path.exists(test_csv):
            return jsonify({'success': False, 'error': 'Valid test_csv required'}), 400

        result = run_keystroke_knn(train_csv=train_csv, test_csv=test_csv, num_bins=num_bins, n_neighbors=n_neighbors)
        return jsonify({'success': True, **result})
    except Exception as e:
        logger.error(f"Keystroke KNN endpoint error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/network-info', methods=['GET'])
def get_network_info():
    """Get network and request information"""
    try:
        return jsonify({
            'success': True,
            'network_info': {
                'ip_address': request.remote_addr,
                'user_agent': request.headers.get('User-Agent', 'Unknown'),
                'timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Network info error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/enhanced-auth')
def enhanced_auth_redirect():
    """Redirect to enhanced authentication page"""
    if not session.get('needs_enhanced_verification'):
        return jsonify({'error': 'Enhanced verification not required'}), 400
    
    # Return HTML page for enhanced authentication
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Enhanced Authentication Required</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .auth-container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                padding: 40px;
                max-width: 600px;
                width: 100%;
                text-align: center;
            }
            .auth-header h1 {
                color: #333;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .auth-header p {
                color: #666;
                font-size: 1.1em;
            }
            .btn {
                background: linear-gradient(45deg, #007bff, #0056b3);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 25px;
                font-size: 1.1em;
                cursor: pointer;
                transition: all 0.3s ease;
                margin: 10px;
                text-decoration: none;
                display: inline-block;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(0,123,255,0.3);
            }
            .security-info {
                background: #e3f2fd;
                border: 1px solid #bbdefb;
                border-radius: 10px;
                padding: 15px;
                margin: 20px 0;
                text-align: left;
            }
        </style>
    </head>
    <body>
        <div class="auth-container">
            <div class="auth-header">
                <h1>🔐 Enhanced Authentication Required</h1>
                <p>Additional verification needed for security</p>
            </div>
            
            <div class="security-info">
                <h4>🛡️ Why Enhanced Verification?</h4>
                <p>Your behavioral patterns didn't match our security requirements. To ensure your account security, we need additional verification through:</p>
                <ul>
                    <li>Face recognition scan</li>
                    <li>Voice pattern analysis</li>
                    <li>Location and device verification</li>
                </ul>
            </div>
            
            <a href="http://localhost:5001/enhanced-auth" class="btn">
                Start Enhanced Verification
            </a>
            
            <p style="margin-top: 20px; color: #666;">
                This process takes about 2-3 minutes and helps protect your account.
            </p>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("Starting Integrated Behavioral Biometric Authentication Server...")
    print("Frontend available at: http://localhost:5000")
    print("API endpoints available at: http://localhost:5000/api/")
    print("Health check: http://localhost:5000/api/health")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
