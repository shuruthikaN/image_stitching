import cv2
import numpy as np
import random

# ================= ORB FEATURE ENGINE =================
class FeatureEngine:
    def __init__(self, max_features=3000):
        self.orb = cv2.ORB_create(max_features)

    def extract(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, None)
        return kp, desc

# ================= HAMMING DISTANCE =================
def hamming_distance(a, b):
    return np.count_nonzero(a ^ b)

# ================= MANUAL ORB MATCH =================
def manual_match(desc1, desc2, ratio=0.75):
    matches = []
    for i in range(len(desc1)):
        best, second, best_idx = 1e9, 1e9, -1
        for j in range(len(desc2)):
            d = hamming_distance(desc1[i], desc2[j])
            if d < best:
                second, best = best, d
                best_idx = j
            elif d < second:
                second = d
        if best < ratio * second:
            matches.append((i, best_idx))
    return matches

# ================= NORMALIZATION =================
def normalize_points(pts):
    pts = np.array(pts)
    c = np.mean(pts, axis=0)
    d = np.mean(np.linalg.norm(pts - c, axis=1))
    s = np.sqrt(2) / d
    T = np.array([[s,0,-s*c[0]],
                  [0,s,-s*c[1]],
                  [0,0,1]])
    pts_h = np.hstack([pts, np.ones((len(pts),1))])
    pts_n = (T @ pts_h.T).T
    return pts_n[:,0:2], T

# ================= DLT HOMOGRAPHY =================
def compute_homography(src, dst):
    src_n, Ts = normalize_points(src)
    dst_n, Td = normalize_points(dst)
    A = []
    for (x,y),(u,v) in zip(src_n, dst_n):
        A.append([-x,-y,-1,0,0,0,x*u,y*u,u])
        A.append([0,0,0,-x,-y,-1,x*v,y*v,v])
    A = np.array(A)
    _,_,Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3,3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2,2]

# ================= RANSAC =================
def ransac(src, dst, iters=2000, thresh=3):
    best_H = None
    max_in = 0
    for _ in range(iters):
        if len(src) < 4:
            return None
        idx = random.sample(range(len(src)), 4)
        H = compute_homography([src[i] for i in idx],
                               [dst[i] for i in idx])
        inliers = 0
        for p, q in zip(src, dst):
            ph = np.array([p[0], p[1], 1.0])
            qh = H @ ph
            if abs(qh[2]) < 1e-6:
                continue
            qh /= qh[2]
            if np.linalg.norm(qh[:2]-q) < thresh:
                inliers += 1
        if inliers > max_in:
            max_in = inliers
            best_H = H
    return best_H

# ================= MANUAL WARP =================
def manual_warp(src, canvas, H):
    Hinv = np.linalg.inv(H)
    h, w = canvas.shape[:2]
    for y in range(h):
        for x in range(w):
            p = np.array([x, y, 1.0])
            s = Hinv @ p
            if abs(s[2]) < 1e-6:
                continue
            sx, sy = int(s[0]/s[2]), int(s[1]/s[2])
            if 0 <= sx < src.shape[1] and 0 <= sy < src.shape[0]:
                canvas[y, x] = src[sy, sx]

# ================= STITCH SINGLE IMAGE =================
def stitch(src_idx, ref_idx, tiles, canvas, Hglob, fe, min_matches=40):
    kp1, d1 = fe.extract(tiles[src_idx])
    kp2, d2 = fe.extract(tiles[ref_idx])
    matches = manual_match(d1, d2)
    print(f"Matches {src_idx}->{ref_idx}: {len(matches)}")
    if len(matches) < min_matches:
        print("Not enough matches. Skipping.")
        return
    pts1 = [kp1[i].pt for i,_ in matches]
    pts2 = [kp2[j].pt for _,j in matches]
    H = ransac(pts1, pts2)
    if H is None:
        print("Homography failed")
        return
    Hglob[src_idx] = Hglob[ref_idx] @ H
    warped = np.zeros_like(canvas)
    manual_warp(tiles[src_idx], warped, Hglob[src_idx])
    mask = np.any(warped != 0, axis=2)
    canvas[mask] = warped[mask]

# ================= STITCH ALL IMAGES =================
def stitch_all(tiles, canvas, Hglob, fe):
    stitch_pairs = [
        (1, 4), (3, 4), (5, 4), (7, 4),
        (0, 1), (2, 1), (6, 3), (8, 5)
    ]
    for src, ref in stitch_pairs:
        stitch(src, ref, tiles, canvas, Hglob, fe)

# ================= MAIN =================
if __name__ == "__main__":

    # Load tiles
    tiles = [cv2.imread(f"tile{i+1}.jpg") for i in range(9)]
    if any(t is None for t in tiles):
        print("Image load failed")
        exit()

    # Feature engine
    fe = FeatureEngine()

    # Create canvas
    Hc = tiles[4].shape[0] * 3
    Wc = tiles[4].shape[1] * 3
    canvas = np.zeros((Hc, Wc, 3), np.uint8)

    # Place center tile (tile 4) in middle of canvas
    cx = Wc//2 - tiles[4].shape[1]//2
    cy = Hc//2 - tiles[4].shape[0]//2
    Hglob = [None]*9
    Hglob[4] = np.array([[1,0,cx],[0,1,cy],[0,0,1]])
    canvas[cy:cy+tiles[4].shape[0], cx:cx+tiles[4].shape[1]] = tiles[4]

    # Stitch all other tiles
    stitch_all(tiles, canvas, Hglob, fe)

    # Display final panorama
    cv2.namedWindow("Final ORB Stitch", cv2.WINDOW_NORMAL)
    cv2.imshow("Final ORB Stitch", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
