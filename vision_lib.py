"""BairroVivo Vision — funções comuns: leitura do dataset YOLO, vazamento entre splits, mapeamento para ocorrências."""
from __future__ import annotations

import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# --------------------------------------------------------------------------------------------- arquivos
def imread(path):
    """cv2.imread que aceita caminho com acento no Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path, img, quality=90):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(Path(path).suffix or '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        buf.tofile(str(path))
    return ok


def achar_data_yaml(raiz):
    raiz = Path(raiz)
    achados = sorted(raiz.rglob('data.yaml'), key=lambda p: len(p.parts))
    if not achados:
        raise FileNotFoundError(f'Nenhum data.yaml dentro de {raiz}. Exporte do Roboflow no formato YOLOv11 e descompacte aí.')
    return achados[0]


def ler_dataset(data_yaml):
    """Lê um export do Roboflow/Ultralytics. Devolve (nomes, {split: [(img, label)]})."""
    import yaml
    data_yaml = Path(data_yaml)
    cfg = yaml.safe_load(open(data_yaml, encoding='utf-8'))
    nomes = cfg['names']
    if isinstance(nomes, dict):
        nomes = [nomes[k] for k in sorted(nomes, key=int)]
    raiz = data_yaml.parent
    splits = {}
    for sp, chave in (('train', 'train'), ('val', 'val'), ('test', 'test')):
        alvo = cfg.get(chave)
        candidatos = [raiz / sp / 'images', raiz / ('valid' if sp == 'val' else sp) / 'images']
        if alvo:
            alvo = str(alvo).replace('../', '')
            candidatos.insert(0, raiz / alvo)
        pasta = next((c for c in candidatos if c.is_dir()), None)
        if pasta is None:
            continue
        itens = []
        for img in sorted(pasta.iterdir()):
            if img.suffix.lower() in IMG_EXT:
                lab = pasta.parent / 'labels' / (img.stem + '.txt')
                itens.append((img, lab))
        splits[sp] = itens
    return list(nomes), splits


def ler_rotulos(lab_path):
    """[(classe, x, y, w, h)] em coordenadas normalizadas. Polígonos (segmentação) viram a caixa envolvente."""
    out = []
    if not Path(lab_path).exists():
        return out
    for linha in Path(lab_path).read_text(encoding='utf-8', errors='ignore').splitlines():
        v = linha.split()
        if len(v) < 5:
            continue
        c, nums = int(float(v[0])), list(map(float, v[1:]))
        if len(nums) == 4:
            x, y, w, h = nums
        else:                                   # polígono: x1 y1 x2 y2 ...
            xs, ys = nums[0::2], nums[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            x, y, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        if w > 0 and h > 0:
            out.append((c, x, y, w, h))
    return out


# --------------------------------------------------------------------------------------------- vazamento
RF_RE = re.compile(r'^(?P<orig>.+?)(?:_(?:jpg|jpeg|png|JPG|JPEG|PNG))?\.rf\.[0-9a-f]{20,}$')


def nome_original(stem):
    """O Roboflow renomeia 'foto.jpg' para 'foto_jpg.rf.<hash>.jpg' e cada cópia aumentada ganha outro hash.
    O nome antes de '.rf.' identifica a foto de origem."""
    m = RF_RE.match(stem)
    return m.group('orig') if m else stem


def dhash(img, size=8):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    r = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    return np.packbits((r[:, 1:] > r[:, :-1]).flatten())


def grupos_por_origem(splits):
    """{nome_original: [(split, img, label)]}"""
    g = defaultdict(list)
    for sp, itens in splits.items():
        for img, lab in itens:
            g[nome_original(img.stem)].append((sp, img, lab))
    return g


def vazamento_por_nome(splits):
    g = grupos_por_origem(splits)
    cruzados = {k: v for k, v in g.items() if len({sp for sp, _, _ in v}) > 1}
    return g, cruzados


def vazamento_por_hash(splits, max_dist=3, amostra=None, seed=0):
    """Conta imagens de val/test quase idênticas (dHash) a alguma imagem de treino."""
    rng = random.Random(seed)
    hashes = {}
    for sp, itens in splits.items():
        sel = itens if amostra is None or len(itens) <= amostra else rng.sample(itens, amostra)
        hs = []
        for img, _ in sel:
            im = imread(img)
            if im is not None:
                hs.append((img, dhash(im)))
        hashes[sp] = hs
    if 'train' not in hashes:
        return {}
    tr = np.unpackbits(np.stack([h for _, h in hashes['train']]), axis=1) if hashes['train'] else None
    res = {}
    for sp in ('val', 'test'):
        if sp not in hashes or tr is None:
            continue
        n = 0
        for _, h in hashes[sp]:
            d = np.count_nonzero(tr != np.unpackbits(h)[None], axis=1)
            n += int(d.min() <= max_dist)
        res[sp] = {'verificadas': len(hashes[sp]), 'quase_iguais_ao_treino': n}
    return res


# --------------------------------------------------------------------------------------------- estatísticas
def estatisticas(nomes, splits):
    st = {}
    for sp, itens in splits.items():
        inst, imgs_cls, vazias = Counter(), Counter(), 0
        for _, lab in itens:
            rot = ler_rotulos(lab)
            if not rot:
                vazias += 1
            for c in {r[0] for r in rot}:
                imgs_cls[c] += 1
            for r in rot:
                inst[r[0]] += 1
        st[sp] = {'imagens': len(itens), 'sem_rotulo': vazias,
                  'por_classe': {nomes[c] if c < len(nomes) else str(c): {'imagens': imgs_cls[c], 'instancias': inst[c]}
                                 for c in sorted(set(inst) | set(imgs_cls))}}
    return st


def grade_exemplos(itens, nomes, classe, n=12, cols=4, lado=256, seed=0):
    """Mosaico com n imagens que contêm a classe, com as caixas desenhadas."""
    rng = random.Random(seed)
    com = [(i, l) for i, l in itens if any(r[0] == classe for r in ler_rotulos(l))]
    sel = rng.sample(com, min(n, len(com)))
    tiles = []
    for img_p, lab in sel:
        im = imread(img_p)
        if im is None:
            continue
        H, W = im.shape[:2]
        for c, x, y, w, h in ler_rotulos(lab):
            cor = (0, 220, 255) if c == classe else (160, 160, 160)
            p0 = (int((x - w / 2) * W), int((y - h / 2) * H)); p1 = (int((x + w / 2) * W), int((y + h / 2) * H))
            cv2.rectangle(im, p0, p1, cor, max(2, W // 200))
        tiles.append(cv2.resize(im, (lado, lado)))
    if not tiles:
        return None
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


# --------------------------------------------------------------------------------------------- dataset final
def amostra_por_cota(itens, limite, n_classes, rng):
    """Escolhe `limite` imagens tentando dar a mesma quantidade de imagens a cada classe.

    itens: [(img, rotulos)]. Começa pela classe mais rara; depois completa com imagens aleatórias.
    """
    idx = list(range(len(itens)))
    rng.shuffle(idx)
    por_classe = defaultdict(list)
    for i in idx:
        for c in {r[0] for r in itens[i][1]}:
            por_classe[c].append(i)
    cota = limite / max(1, n_classes)
    escolhidos, cont = set(), Counter()
    for c in sorted(por_classe, key=lambda k: len(por_classe[k])):
        for i in por_classe[c]:
            if cont[c] >= cota or len(escolhidos) >= limite:
                break
            if i not in escolhidos:
                escolhidos.add(i)
                for cc in {r[0] for r in itens[i][1]}:
                    cont[cc] += 1
    for i in idx:
        if len(escolhidos) >= limite:
            break
        escolhidos.add(i)
    return [itens[i] for i in sorted(escolhidos)]


def montar_dataset(nomes, splits, destino, classes_mantidas, max_por_split, seed=42, fracs=(0.8, 0.1, 0.1),
                   negativos_frac=0.05):
    """Recria treino/val/test SEM vazamento: agrupa as cópias de cada foto de origem e sorteia o split por grupo.

    classes_mantidas: lista de nomes originais a manter (as outras são removidas dos rótulos).
    max_por_split: {'train': n, 'val': n, 'test': n} — limita o tamanho (o dataset original é grande para uma RTX 3050).
    Dentro de cada grupo usa só 1 imagem (a primeira), para não repetir cópias aumentadas.
    """
    rng = random.Random(seed)
    destino = Path(destino)
    if destino.exists():
        shutil.rmtree(destino)
    novo_id = {nomes.index(c): i for i, c in enumerate(classes_mantidas)}
    grupos = grupos_por_origem(splits)
    chaves = sorted(grupos)
    rng.shuffle(chaves)
    n = len(chaves)
    cortes = (int(n * fracs[0]), int(n * (fracs[0] + fracs[1])))
    atribui = {k: ('train' if i < cortes[0] else 'val' if i < cortes[1] else 'test') for i, k in enumerate(chaves)}

    # equilibra: percorre os grupos priorizando a classe mais rara já escolhida
    por_split = {'train': [], 'val': [], 'test': []}
    for k in chaves:
        sp_new = atribui[k]
        _, img, lab = sorted(grupos[k], key=lambda t: str(t[1]))[0]
        rot = [(novo_id[c], x, y, w, h) for c, x, y, w, h in ler_rotulos(lab) if c in novo_id]
        por_split[sp_new].append((img, rot))

    stats = {}
    for sp, itens in por_split.items():
        com = [it for it in itens if it[1]]
        sem = [it for it in itens if not it[1]]
        limite = max_por_split.get(sp)
        if limite and len(com) > limite:
            com = amostra_por_cota(com, limite, len(classes_mantidas), rng)
        sem = rng.sample(sem, min(len(sem), int(len(com) * negativos_frac)))
        sel = com + sem
        (destino / 'images' / sp).mkdir(parents=True, exist_ok=True)
        (destino / 'labels' / sp).mkdir(parents=True, exist_ok=True)
        cont = Counter()
        for i, (img, rot) in enumerate(sel):
            nome = f'{sp}_{i:05d}{img.suffix.lower()}'
            shutil.copy2(img, destino / 'images' / sp / nome)
            (destino / 'labels' / sp / (Path(nome).stem + '.txt')).write_text(
                ''.join(f'{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n' for c, x, y, w, h in rot))
            for c, *_ in rot:
                cont[classes_mantidas[c]] += 1
        stats[sp] = {'imagens': len(sel), 'negativas': len(sem), 'instancias': dict(cont)}
    import yaml
    (destino / 'data.yaml').write_text(yaml.safe_dump({'path': str(destino), 'train': 'images/train', 'val': 'images/val',
                                                       'test': 'images/test', 'names': dict(enumerate(classes_mantidas))},
                                                      allow_unicode=True, sort_keys=False), encoding='utf-8')
    return stats


# --------------------------------------------------------------------------------------------- BairroVivo
# Mapeia o nome da classe do dataset para a categoria de ocorrência do BairroVivo (por palavra-chave).
CATEGORIAS = [
    # (palavras-chave, nome em português, categoria BairroVivo, prioridade base)
    (('pothole', 'buraco'), 'Buraco', 'Buraco / pavimento', 'alta'),
    (('sign',), 'Placa danificada', 'Mobiliário urbano', 'média'),   # antes de 'road': Damaged_Road_Signs é placa
    (('road', 'crack', 'pavement', 'asphalt'), 'Via danificada', 'Buraco / pavimento', 'média'),
    (('garbage', 'trash', 'litter', 'waste', 'dump', 'lixo'), 'Lixo / descarte irregular', 'Descarte irregular', 'média'),
    (('electric', 'pole', 'wire', 'poste'), 'Poste ou fiação danificada', 'Iluminação / rede elétrica', 'alta'),
    (('concrete', 'structure', 'sidewalk', 'calcada'), 'Estrutura de concreto danificada', 'Calçada / acessibilidade', 'média'),
    (('graffiti', 'vandal', 'pichacao'), 'Pichação / vandalismo', 'Mobiliário urbano', 'baixa'),
    (('tree',), 'Árvore caída', 'Via obstruída', 'alta'),
    (('parking',), 'Estacionamento irregular', 'Via obstruída', 'baixa'),
    (('animal',), 'Animal morto', 'Descarte irregular', 'média'),
]


def categoria(nome_classe):
    n = nome_classe.lower().replace('-', '_')
    for chaves, pt, cat, prio in CATEGORIAS:
        if any(k in n for k in chaves):
            return {'classe': nome_classe, 'nome_pt': pt, 'categoria': cat, 'prioridade': prio}
    return {'classe': nome_classe, 'nome_pt': nome_classe.replace('_', ' ').capitalize(), 'categoria': 'Outros',
            'prioridade': 'média'}


CORES_PRIORIDADE = {'alta': (230, 70, 60), 'média': (240, 180, 40), 'baixa': (80, 170, 230)}  # RGB
