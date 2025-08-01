import numpy as np
from typing import List, Dict, Tuple
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
        if color not in self._color_map:
            self._color_map[color] = len(self._color_map)
        return self._color_map[color]

    def compute_visual_weight(self, style: Dict) -> float:
        weight = style["fontsize"]
        if style.get("bold") == -1:
            weight += 20
        if style.get("italic") == -1:
            weight += 5
        return weight

    def cluster_styles(
        self,
        styles: List[Dict],
        frame_style_map: Dict[str, List[str]]
    ) -> Tuple[List[Dict], Dict[str, List[str]]]:

        style_entries = []
        style_features = []
        seen_keys = set()
        style_to_word_refs = defaultdict(list)
        key_to_entry = {}
        key_to_index = {}

        # Step 1: Deduplicate and collect features
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

            key_to_index[key] = len(style_entries)
            style_entries.append(word)
            style_features.append(vec)
            key_to_entry[key] = word

        if not style_features:
            return [], frame_style_map

        # Step 2: Normalize and cluster
        features = np.array(style_features)
        scaled = MinMaxScaler().fit_transform(features)

        dbscan = DBSCAN(eps=0.3, min_samples=1)
        labels = dbscan.fit_predict(scaled)

        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append((idx, scaled[idx]))

        representatives = []
        cluster_keys = []
        key_to_rep = {}

        # Step 3: Find representatives for clusters
        for label, group in clusters.items():
            center = np.mean([g[1] for g in group], axis=0)
            closest_idx = min(group, key=lambda g: np.linalg.norm(g[1] - center))[0]
            rep_word = style_entries[closest_idx]

            rep_key = (
                rep_word["fontname"],
                rep_word["fontsize"],
                rep_word["primary_colour"],
                rep_word["bold"],
                rep_word["italic"]
            )

            representatives.append(rep_word)
            keys_in_cluster = []
            for idx, _ in group:
                original_word = style_entries[idx]
                key = (
                    original_word["fontname"],
                    original_word["fontsize"],
                    original_word["primary_colour"],
                    original_word["bold"],
                    original_word["italic"]
                )
                key_to_rep[key] = rep_word
                keys_in_cluster.append(key)

            cluster_keys.append(keys_in_cluster)

        # Step 4: Compute scores
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

        # Step 5: Rank styles
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
        print("cluster logic chal rha h ")
        print(ranked_styles)
        # Step 6: Map original style names to ranked styles
        original_to_clustered = {}

        for style in styles:
            original_name = style["name"]
            key = (
                style["fontname"],
                style["fontsize"],
                style["primary_colour"],
                style["bold"],
                style["italic"]
            )
            if key in key_to_rep:
                rep = key_to_rep[key]
                # Find ranked name of this representative
                for ranked_style in ranked_styles:
                    if (ranked_style["fontname"] == rep["fontname"] and
                        ranked_style["fontsize"] == rep["fontsize"] and
                        ranked_style["primary_colour"] == rep["primary_colour"] and
                        ranked_style["bold"] == rep["bold"] and
                        ranked_style["italic"] == rep["italic"]):
                        original_to_clustered[original_name] = ranked_style["name"]
                        break
            else:
                original_to_clustered[original_name] = "style1" if ranked_styles else "default"

        # Step 7: Update frame_style_map
        updated_frame_style_map = {}
        for frame_id, style_names in frame_style_map.items():
            updated_frame_style_map[frame_id] = [
                original_to_clustered.get(style_name, style_name)
                for style_name in style_names
            ]
        print('mapping bhi chal rhi h ')
        return ranked_styles, updated_frame_style_map
