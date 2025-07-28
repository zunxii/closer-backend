import numpy as np
from typing import List, Dict
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from config import Config


class StyleClusterer:
    def __init__(self):
        self.alpha = Config.ALPHA
        self.beta = Config.BETA
        self.min_frequency = Config.MIN_FREQUENCY
        self._color_map = {}

    def _color_string_to_index(self, color: str) -> int:
        """Convert a unique color string like '&HFF00FF' into an index."""
        if color not in self._color_map:
            self._color_map[color] = len(self._color_map)
        return self._color_map[color]

    def compute_visual_weight(self, style: Dict) -> float:
        """Compute visual weight of a style"""
        weight = style["fontsize"]
        if style.get("bold") == -1:
            weight += 20
        if style.get("italic") == -1:
            weight += 5
        return weight

    def cluster_styles(self, styles: List[Dict]) -> List[Dict]:
        """Cluster similar flat style dicts and return ranked representatives"""
        style_entries = []
        style_features = []
        seen_keys = set()
        style_to_word_refs = defaultdict(list)

        # Collect unique styles
        for word in styles:
            key = (
                word["fontname"],
                word["fontsize"],
                word["primary_colour"],
                word["bold"],
                word["italic"]
            )
            style_to_word_refs[key].append(word)

            if key in seen_keys:
                continue
            seen_keys.add(key)

            vec = [
                word["fontsize"],
                int(word["bold"] != 0),
                int(word["italic"] != 0),
                self._color_string_to_index(word["primary_colour"])
            ]

            style_entries.append(word)
            style_features.append(vec)

        if not style_features:
            return []

        # Cluster styles
        features = np.array(style_features)
        scaled = MinMaxScaler().fit_transform(features)

        dbscan = DBSCAN(eps=0.3, min_samples=1)
        labels = dbscan.fit_predict(scaled)

        # Find cluster representatives
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append((idx, scaled[idx]))

        representatives = []
        cluster_keys = []

        for label, group in clusters.items():
            center = np.mean([g[1] for g in group], axis=0)
            closest_idx = min(group, key=lambda g: np.linalg.norm(g[1] - center))[0]
            rep_word = style_entries[closest_idx]

            representatives.append(rep_word)
            cluster_keys.append([(
                style_entries[idx]["fontname"],
                style_entries[idx]["fontsize"],
                style_entries[idx]["primary_colour"],
                style_entries[idx]["bold"],
                style_entries[idx]["italic"]
            ) for idx, _ in group])

        # Calculate style scores
        style_counts = []
        for keys in cluster_keys:
            count = sum(len(style_to_word_refs[k]) for k in keys)
            style_counts.append(count)

        max_count = max(style_counts) if style_counts else 1
        scored_styles = []

        for word, count in zip(representatives, style_counts):
            if count < self.min_frequency:
                continue

            visual_weight = self.compute_visual_weight(word)
            frequency_weight = count / max_count
            total_score = self.alpha * visual_weight + self.beta * frequency_weight * 100

            word = word.copy()
            word["score"] = round(total_score, 2)
            scored_styles.append(word)

        # Sort and format output
        scored_styles.sort(key=lambda w: w["score"])

        ranked_styles = []
        for i, style in enumerate(scored_styles):
            ranked_styles.append({
                "name": f"style{i+1}",
                "fontname": style["fontname"],
                "fontsize": style["fontsize"],
                "primary_colour": style["primary_colour"],
                "bold": style["bold"],
                "italic": style["italic"],
                "outline": style.get("outline", 0),
                "shadow": style.get("shadow", 0),
                "frame_id": style.get("frame_id", "")
            })
        print(ranked_styles)
        return ranked_styles
