"""Reference-only XML adapter and auditable custom reader consolidation.

Never import this module from the annotation-free predictor. Real XML coverage
and reader association require a development audit before cohort release.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.distance import pdist
from skimage.draw import polygon
from .geometry import to_world, to_index


def xml_root(path):
    # The adapter accepts data, not DTD/entity programs.
    raw = Path(path).read_bytes()
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("DTD/entities not supported")
    root = ET.fromstring(raw)
    for elem in root.iter():
        elem.tag = elem.tag.split("}")[-1]
    return root


def parse_xml(path, volume):
    root = xml_root(path)
    series = root.findtext(".//SeriesInstanceUid") or root.findtext(".//SeriesInstanceUID")
    if series != volume.metadata["series_uid"]:
        raise ValueError("XML/CT series linkage unresolved")
    sop_to_z = {uid: z for z, uid in enumerate(volume.metadata["sop_uids"])}
    marks, readers = [], set()
    sessions = root.findall(".//readingSession")
    if not sessions:
        raise ValueError("No final reading sessions")
    for session_no, session in enumerate(sessions):
        reader = session.findtext("servicingRadiologistID")
        if not reader:
            raise ValueError("Reader identity missing; resolve explicitly before release")
        if reader in readers:
            raise ValueError("Duplicate reader session")
        readers.add(reader)
        unknown = {e.tag for e in session} - {"servicingRadiologistID", "unblindedReadNodule", "nonNodule"}
        if unknown:
            raise ValueError(f"Unsupported reading-session fields: {sorted(unknown)}")
        for number, node in enumerate(session.findall("unblindedReadNodule")):
            identifier = node.findtext("noduleID")
            if not identifier:
                raise ValueError("Missing nodule ID")
            per_slice, holes, point_marks, diameters = {}, {}, [], []
            for roi in node.findall("roi"):
                uid = roi.findtext("imageSOP_UID")
                if uid not in sop_to_z:
                    raise ValueError("ROI SOP not in selected series")
                z = sop_to_z[uid]
                stated_z = float(roi.findtext("imageZposition"))
                if abs(stated_z - volume.metadata["slice_positions"][z][2]) > 0.1:
                    raise ValueError("XML z coordinate disagrees with SOP geometry")
                xy = np.array([[float(e.findtext("xCoord")), float(e.findtext("yCoord"))]
                               for e in roi.findall("edgeMap")])
                if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
                    raise ValueError("Invalid contour coordinates")
                if (xy < 0).any() or (xy[:, 0] >= volume.hu.shape[2]).any() or (xy[:, 1] >= volume.hu.shape[1]).any():
                    raise ValueError("Contour outside source grid")
                inclusion = roi.findtext("inclusion", "").upper()
                if inclusion not in ("TRUE", "FALSE"):
                    raise ValueError("Contour inclusion flag missing")
                if len(xy) < 3:
                    point_marks.extend([[z, y, x] for x, y in xy])
                    continue
                target = per_slice if inclusion == "TRUE" else holes
                plane = target.setdefault(z, np.zeros(volume.hu.shape[1:], bool))
                rr, cc = polygon(xy[:, 1], xy[:, 0], shape=plane.shape)
                plane[rr, cc] = True
                if inclusion == "TRUE":
                    points = to_world(np.column_stack([np.full(len(xy), z), xy[:, 1], xy[:, 0]]), volume.affine)
                    diameters.append(float(pdist(points).max()))
            coords = []
            for z, mask in per_slice.items():
                mask &= ~holes.get(z, np.zeros_like(mask))
                yx = np.argwhere(mask)
                coords.extend(np.column_stack([np.full(len(yx), z), yx]).tolist())
            if coords and point_marks:
                raise ValueError("Mixed point/contour nodule requires reference review")
            if coords:
                coords = np.unique(np.asarray(coords, int), axis=0)
                center = to_world(coords, volume.affine).mean(0)
                diameter = max(diameters)
            elif point_marks:
                coords = np.empty((0, 3), int)
                center = to_world(point_marks, volume.affine).mean(0)
                diameter = None
            else:
                raise ValueError("Empty nodule annotation")
            marks.append({"id": f"{reader}:{identifier}", "reader": reader, "kind": "nodule",
                          "center_lps": center.tolist(), "diameter_mm": diameter,
                          "coords": coords})
        for number, node in enumerate(session.findall("nonNodule")):
            uid = node.findtext("imageSOP_UID")
            if uid not in sop_to_z:
                raise ValueError("Non-nodule SOP linkage missing")
            x, y = float(node.findtext("locus/xCoord")), float(node.findtext("locus/yCoord"))
            center = to_world([sop_to_z[uid], y, x], volume.affine)
            marks.append({"id": f"{reader}:non:{node.findtext('nonNoduleID', str(number))}",
                "reader": reader, "kind": "non_nodule", "center_lps": center.tolist(),
                "diameter_mm": None, "coords": np.empty((0, 3), int)})
    if len({m["id"] for m in marks}) != len(marks):
        raise ValueError("Duplicate annotation IDs")
    return marks


def consolidate(marks):
    eligible = [m for m in marks if m["kind"] == "nodule" and m["diameter_mm"] is not None]
    eligible.sort(key=lambda m: m["id"])
    clusters = [[m] for m in eligible]
    trace, ambiguity = [], []
    def merge_distance(a, b):
        if set(m["reader"] for m in a) & set(m["reader"] for m in b):
            return None
        ratios = [np.linalg.norm(np.array(x["center_lps"])-y["center_lps"]) /
                  ((x["diameter_mm"]+y["diameter_mm"])/2) for x in a for y in b]
        return max(ratios) if max(ratios) < 1 else None
    # Competing same-reader associations are surfaced even if a greedy merge
    # would otherwise hide the ambiguity. Review decisions remain external.
    for a in eligible:
        neighbors = {}
        for b in eligible:
            if a is not b and merge_distance([a], [b]) is not None:
                neighbors.setdefault(b["reader"], []).append(b["id"])
        for reader, ids in neighbors.items():
            if len(ids) > 1:
                ambiguity.append({"annotation": a["id"], "competing_reader": reader, "choices": sorted(ids)})
    while True:
        options = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = merge_distance(clusters[i], clusters[j])
                if distance is not None:
                    ids = tuple(sorted(m["id"] for m in clusters[i]+clusters[j]))
                    options.append((distance, ids, i, j))
        if not options:
            break
        distance, ids, i, j = min(options)
        trace.append({"members": list(ids), "maximum_pair_ratio": float(distance)})
        clusters[i] = sorted(clusters[i]+clusters[j], key=lambda m: m["id"])
        del clusters[j]
    return clusters, trace, ambiguity


def make_references(marks, affine, mask_dir, scan_id):
    """Emit draft references; any ambiguity requires a signed review record."""
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=False)
    clusters, trace, ambiguities = consolidate(marks)
    positives, ignored, represented = [], [], set()
    def save_mask(identifier, coords):
        if len(coords) == 0:
            return None
        path = mask_dir / f"{identifier}.npz"
        np.savez_compressed(path, coords_zyx=np.asarray(coords, np.int32), affine=affine)
        return str(path.resolve())
    for i, cluster in enumerate(clusters):
        identifier = f"{scan_id}:lesion:{i:04d}"
        represented.update(m["id"] for m in cluster)
        diameter = float(np.mean([m["diameter_mm"] for m in cluster]))
        all_coords = np.concatenate([m["coords"] for m in cluster])
        unique, votes = np.unique(all_coords, axis=0, return_counts=True)
        included = len(cluster) >= 3 and 3 <= diameter <= 30
        coords = unique[votes > len(cluster)/2] if included else unique
        entry = {"id": identifier, "center_lps": np.mean([m["center_lps"] for m in cluster], axis=0).tolist(),
            "radius_mm": diameter/2, "diameter_mm": diameter,
            "reader_ids": [m["reader"] for m in cluster], "member_ids": [m["id"] for m in cluster],
            "mask_path": save_mask(f"cluster_{i:04d}", coords), "point_only": False}
        (positives if included else ignored).append(entry)
    for i, mark in enumerate(marks):
        if mark["id"] in represented:
            continue
        ignored.append({"id": f"{scan_id}:ignore:{i:04d}", "center_lps": mark["center_lps"],
            "radius_mm": 3.0, "point_only": True, "mask_path": None})
    audit = {"merge_trace": trace, "ambiguities": ambiguities,
             "reader_marks": [{k:v for k,v in m.items() if k != "coords"} for m in marks],
             "status": "DRAFT_REQUIRES_REFERENCE_AUDIT"}
    # Retain every contributing reader mask, including disagreement.
    for i, mark in enumerate(marks):
        audit["reader_marks"][i]["mask_path"] = save_mask(f"reader_{i:04d}", mark["coords"])
    return positives, ignored, audit


@lru_cache(maxsize=128)
def mask_lookup(path):
    data = np.load(path, allow_pickle=False)
    return set(map(tuple, data["coords_zyx"].tolist())), data["affine"]


def inside_ignore(center, ignored, point_radius=None):
    for region in ignored:
        radius = point_radius if point_radius is not None and region.get("point_only") else region["radius_mm"]
        distance = np.linalg.norm(np.asarray(center)-region["center_lps"])
        if distance < radius if region.get("strict_radius") else distance <= radius:
            return True
        if region.get("mask_path"):
            coords, affine = mask_lookup(region["mask_path"])
            voxel = tuple(np.floor(to_index(center, affine)+0.5).astype(int))
            if voxel in coords:
                return True
    return False


def candidate_label(center, positive, ignored):
    def contained(region):
        distance = np.linalg.norm(np.asarray(center)-region["center_lps"])
        return distance < region["radius_mm"] if region.get("strict_radius") else distance <= region["radius_mm"]
    if any(contained(region) for region in positive):
        return 1
    return -1 if inside_ignore(center, ignored) else 0
