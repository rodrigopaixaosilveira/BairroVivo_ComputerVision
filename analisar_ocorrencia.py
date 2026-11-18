"""BairroVivo Vision — analisa a foto de uma ocorrência e sugere categoria e prioridade.

É a função que o backend do BairroVivo chama quando o morador envia uma foto. A resposta é uma SUGESTÃO:
o operador sempre pode corrigir. Com confiança baixa ou nada detectado, pede revisão manual.

Uso:
    python analisar_ocorrencia.py foto.jpg --modelo modelos/bairrovivo_vision.pt [--salvar saida.jpg]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_lib import CORES_PRIORIDADE, categoria, imread, imwrite  # noqa: E402

ORDEM_PRIORIDADE = {'alta': 0, 'média': 1, 'baixa': 2}


def montar_resposta(deteccoes, conf_revisao=0.50):
    """deteccoes: [(nome_classe, confianca, [x0, y0, x1, y1])] → dicionário da sugestão."""
    itens = []
    for nome, conf, caixa in deteccoes:
        c = categoria(nome)
        c.update(confianca=round(float(conf), 3), caixa_xyxy=[round(float(v), 1) for v in caixa])
        itens.append(c)
    itens.sort(key=lambda d: (ORDEM_PRIORIDADE[d['prioridade']], -d['confianca']))
    if not itens:
        return {'sugestao': None, 'deteccoes': [], 'revisao_manual': True,
                'motivo': 'nenhum problema reconhecido na imagem'}
    principal = max(itens, key=lambda d: d['confianca'])
    sug = {k: principal[k] for k in ('categoria', 'nome_pt', 'prioridade', 'confianca')}
    # outras categorias presentes na mesma foto (o operador pode abrir mais de uma ocorrência)
    sug['outras_categorias'] = sorted({d['categoria'] for d in itens if d['categoria'] != principal['categoria']
                                       and d['confianca'] >= conf_revisao})
    revisao = principal['confianca'] < conf_revisao
    return {'sugestao': sug, 'deteccoes': itens, 'revisao_manual': revisao,
            'motivo': 'confiança baixa' if revisao else 'ok'}


def analisar_imagem(caminho, modelo, conf_min=0.25, conf_revisao=0.50, device=None):
    """modelo: caminho do .pt ou um objeto YOLO já carregado."""
    from ultralytics import YOLO
    m = YOLO(str(modelo)) if isinstance(modelo, (str, Path)) else modelo
    img = imread(caminho)
    if img is None:
        return {'sugestao': None, 'deteccoes': [], 'revisao_manual': True, 'motivo': 'imagem ilegível'}
    r = m.predict(img, conf=conf_min, verbose=False, device=device)[0]
    dets = [(r.names[int(c)], float(p), b) for c, p, b in
            zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
    return montar_resposta(dets, conf_revisao)


def desenhar(img, resposta):
    """Desenha as caixas e o nome em português (com acento, via PIL)."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    out = img.copy()
    for d in resposta['deteccoes']:
        x0, y0, x1, y1 = map(int, d['caixa_xyxy'])
        cv2.rectangle(out, (x0, y0), (x1, y1), CORES_PRIORIDADE[d['prioridade']][::-1], 3)
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    tam = max(14, img.shape[1] // 40)
    fonte = None
    for nome in ('segoeuib.ttf', 'arialbd.ttf', 'DejaVuSans-Bold.ttf'):
        try:
            fonte = ImageFont.truetype(nome, tam); break
        except OSError:
            continue
    fonte = fonte or ImageFont.load_default()
    for d in resposta['deteccoes']:
        x0, y0 = int(d['caixa_xyxy'][0]), int(d['caixa_xyxy'][1])
        txt = f"{d['nome_pt']} {d['confianca'] * 100:.0f}%"
        l, t, r, b = dr.textbbox((0, 0), txt, font=fonte)
        yy = max(0, y0 - (b - t) - 8)
        dr.rectangle((x0, yy, x0 + (r - l) + 8, yy + (b - t) + 8), fill=CORES_PRIORIDADE[d['prioridade']])
        dr.text((x0 + 4, yy + 2 - t), txt, font=fonte, fill=(20, 20, 20))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('imagem')
    ap.add_argument('--modelo', required=True)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--salvar', default=None, help='grava a imagem com as caixas')
    a = ap.parse_args(argv)
    resp = analisar_imagem(a.imagem, a.modelo, a.conf)
    print(json.dumps(resp, ensure_ascii=False, indent=2))
    if a.salvar:
        imwrite(a.salvar, desenhar(imread(a.imagem), resp))


if __name__ == '__main__':
    main()
