"""Monta um vídeo de demonstração com fotos do conjunto de TESTE, simulando fotos de ocorrências enviadas por
moradores ao BairroVivo, uma atrás da outra. As fotos são sorteadas (não escolhidas a dedo), ~igual por classe.

Uso:
    python montar_demo_fotos.py --dataset dataset/final --saida videos/demo_fotos_moradores.mp4 --n 60 --seg 2
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_lib import imread, ler_rotulos  # noqa: E402

W, H = 1280, 720


def sortear(dataset, n, seed=7):
    """n imagens de teste, divididas igualmente entre as classes presentes (sorteio, sem escolher a dedo)."""
    rng = random.Random(seed)
    img_dir, lab_dir = Path(dataset) / 'images' / 'test', Path(dataset) / 'labels' / 'test'
    por_classe = defaultdict(list)
    for img in sorted(img_dir.glob('*')):
        rot = ler_rotulos(lab_dir / (img.stem + '.txt'))
        if rot:
            por_classe[rot[0][0]].append(img)
    classes = sorted(por_classe)
    por = max(1, n // max(1, len(classes)))
    escolhidas = []
    for c in classes:
        escolhidas += rng.sample(por_classe[c], min(por, len(por_classe[c])))
    rng.shuffle(escolhidas)
    return escolhidas[:n]


def encaixar(img):
    """Redimensiona mantendo a proporção, com fundo escuro, para 1280x720."""
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    nova = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    tela = np.full((H, W, 3), 30, np.uint8)
    y, x = (H - nova.shape[0]) // 2, (W - nova.shape[1]) // 2
    tela[y:y + nova.shape[0], x:x + nova.shape[1]] = nova
    return tela


def montar(dataset, saida, n=60, seg=2.0, fps=30, seed=7):
    fotos = sortear(dataset, n, seed)
    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    wr = None
    for cc in ('mp4v', 'avc1'):
        wr = cv2.VideoWriter(str(saida), cv2.VideoWriter_fourcc(*cc), fps, (W, H))
        if wr.isOpened():
            break
    for i, p in enumerate(fotos, 1):
        im = imread(p)
        if im is None:
            continue
        quadro = encaixar(im)
        cv2.rectangle(quadro, (0, 0), (330, 34), (0, 0, 0), -1)
        cv2.putText(quadro, f'Foto enviada #{i:02d} (simulacao)', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        for _ in range(int(seg * fps)):
            wr.write(quadro)
    wr.release()
    (saida.with_suffix('.txt')).write_text('\n'.join(p.name for p in fotos), encoding='utf-8')
    return fotos


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='dataset/final')
    ap.add_argument('--saida', default='videos/demo_fotos_moradores.mp4')
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--seg', type=float, default=2.0)
    a = ap.parse_args(argv)
    f = montar(a.dataset, a.saida, a.n, a.seg)
    print(f'{len(f)} fotos → {a.saida}')


if __name__ == '__main__':
    main()
