"""BairroVivo Vision — app que roda o modelo nas fotos da pasta `fotos_a_serem_testadas/`.

Cada foto simula uma ocorrência enviada por um morador. Para cada uma, o app:
  1. detecta os problemas com o YOLO11n treinado (modelos/bairrovivo_vision.pt);
  2. monta a sugestão de ocorrência do BairroVivo (categoria, prioridade, confiança, revisão manual);
  3. compara com o rótulo real (`gabarito.csv`, quando existe) e diz se acertou;
  4. mostra a foto com as caixas e um painel lateral, e grava tudo em `saidas/fotos_testadas/`.
As cabeças de pessoas são borradas antes de exibir/salvar (LGPD), como no app de vídeo.

Uso (na pasta do projeto):
    python app_fotos.py                      # abre a janela e passa sozinho, 2 s por foto; Q ou Esc sai
    python app_fotos.py --segundos 3         # outro tempo por foto
    python app_fotos.py --sem-janela         # só processa e grava os resultados
    python app_fotos.py --pasta minhas_fotos # roda em outra pasta (qualquer .jpg/.png; o gabarito é opcional)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from analisar_ocorrencia import desenhar, montar_resposta  # noqa: E402
from app_vision import borrar_cabecas  # noqa: E402
from vision_lib import CORES_PRIORIDADE, imread, imwrite  # noqa: E402

EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
LARG, ALT, PAINEL = 1280, 720, 400          # tela do app: foto à esquerda, painel de 400 px à direita
COR_RES = {'categoria certa': (60, 170, 90), 'só categoria errada': (220, 70, 60), 'nada detectado': (150, 150, 150)}


def fonte(tam, negrito=False):
    nomes = (['segoeuib.ttf', 'arialbd.ttf', 'DejaVuSans-Bold.ttf'] if negrito else
             ['segoeui.ttf', 'arial.ttf', 'DejaVuSans.ttf'])
    for n in nomes:
        try:
            return ImageFont.truetype(n, tam)
        except OSError:
            continue
    return ImageFont.load_default()


def ler_gabarito(pasta):
    arq = Path(pasta) / 'gabarito.csv'
    if not arq.exists():
        return {}
    with open(arq, encoding='utf-8') as f:
        return {l['foto']: l for l in csv.DictReader(f)}


def avaliar(resp, gab, conf):
    """Mesmo critério da demo em vídeo: certa se alguma classe detectada (conf >= limiar) está no rótulo real."""
    if not gab:
        return None
    reais = set(filter(None, gab['classes_reais'].split(';')))
    detect = {d['classe'] for d in resp['deteccoes'] if d['confianca'] >= conf}
    if not detect:
        return 'nada detectado'
    return 'categoria certa' if detect & reais else 'só categoria errada'


def painel(foto_bgr, idx, total, nome, resp, gab, resultado, placar, segundos=2.0):
    """Tela 1280x720: foto com as caixas (esquerda) + painel da ocorrência (direita)."""
    tela = Image.new('RGB', (LARG, ALT), (24, 28, 32))
    h, w = foto_bgr.shape[:2]
    s = min((LARG - PAINEL) / w, ALT / h)
    nova = cv2.resize(foto_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    tela.paste(Image.fromarray(cv2.cvtColor(nova, cv2.COLOR_BGR2RGB)),
               ((LARG - PAINEL - nova.shape[1]) // 2, (ALT - nova.shape[0]) // 2))
    d = ImageDraw.Draw(tela)
    x0 = LARG - PAINEL
    d.rectangle((x0, 0, LARG, ALT), fill=(250, 250, 247))
    x, y = x0 + 22, 18
    d.text((x, y), 'BairroVivo · ocorrência recebida', font=fonte(17, True), fill=(31, 78, 121)); y += 30
    d.text((x, y), f'Foto {idx} de {total}', font=fonte(15), fill=(90, 90, 90)); y += 22
    d.text((x, y), nome[:44], font=fonte(13), fill=(120, 120, 120)); y += 34

    sug = resp['sugestao']
    d.text((x, y), 'SUGESTÃO DO MODELO', font=fonte(13, True), fill=(90, 90, 90)); y += 22
    if sug:
        d.text((x, y), sug['nome_pt'], font=fonte(22, True), fill=(20, 20, 20)); y += 32
        d.text((x, y), sug['categoria'], font=fonte(15), fill=(60, 60, 60)); y += 28
        cor = CORES_PRIORIDADE[sug['prioridade']]
        d.rounded_rectangle((x, y, x + 150, y + 28), 6, fill=cor)
        d.text((x + 10, y + 4), f"prioridade {sug['prioridade']}", font=fonte(14, True), fill=(20, 20, 20))
        d.text((x + 168, y + 4), f"confiança {sug['confianca'] * 100:.0f}%", font=fonte(15), fill=(40, 40, 40)); y += 42
        if sug['outras_categorias']:
            d.text((x, y), 'Também na foto: ' + ', '.join(sug['outras_categorias'])[:48], font=fonte(13),
                   fill=(60, 60, 60)); y += 24
    else:
        d.text((x, y), 'Nenhum problema reconhecido', font=fonte(19, True), fill=(20, 20, 20)); y += 34
    rev = resp['revisao_manual']
    d.text((x, y), ('→ vai para REVISÃO MANUAL' + (' (confiança baixa)' if sug else '')) if rev
           else '→ ocorrência pré-classificada',
           font=fonte(14, True), fill=(200, 110, 0) if rev else (40, 130, 70)); y += 34

    d.text((x, y), f"DETECÇÕES ({len(resp['deteccoes'])})", font=fonte(13, True), fill=(90, 90, 90)); y += 22
    for det in resp['deteccoes'][:5]:
        d.text((x, y), f"• {det['nome_pt']} — {det['confianca'] * 100:.0f}%", font=fonte(14), fill=(40, 40, 40)); y += 21
    y += 14
    if gab is not None:
        d.line((x, y, LARG - 22, y), fill=(210, 210, 210), width=1); y += 14
        d.text((x, y), 'RÓTULO REAL (gabarito)', font=fonte(13, True), fill=(90, 90, 90)); y += 22
        d.text((x, y), gab['nomes_pt'][:46] or '—', font=fonte(15), fill=(20, 20, 20)); y += 28
        if resultado:
            d.rounded_rectangle((x, y, LARG - 22, y + 30), 6, fill=COR_RES[resultado])
            txt = {'categoria certa': 'ACERTOU a categoria', 'só categoria errada': 'ERROU a categoria',
                   'nada detectado': 'NADA DETECTADO (vai para revisão)'}[resultado]
            d.text((x + 10, y + 5), txt, font=fonte(15, True), fill=(255, 255, 255)); y += 44
    # placar acumulado
    yb = ALT - 92
    d.line((x, yb, LARG - 22, yb), fill=(210, 210, 210), width=1)
    d.text((x, yb + 10), 'PLACAR ATÉ AQUI', font=fonte(13, True), fill=(90, 90, 90))
    d.text((x, yb + 32), f"certas {placar['categoria certa']} · erradas {placar['só categoria errada']} · "
                          f"sem detecção {placar['nada detectado']}", font=fonte(14), fill=(40, 40, 40))
    d.text((x, yb + 56), f'passa sozinho a cada {segundos:g} s · Q ou Esc sai', font=fonte(12), fill=(130, 130, 130))
    return cv2.cvtColor(np.array(tela), cv2.COLOR_RGB2BGR)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pasta', default=str(AQUI / 'fotos_a_serem_testadas'))
    ap.add_argument('--modelo', default=str(AQUI / 'modelos' / 'bairrovivo_vision.pt'))
    ap.add_argument('--conf', type=float, default=0.35, help='confiança mínima de uma detecção (padrão 0,35)')
    ap.add_argument('--revisao', type=float, default=0.50, help='abaixo disso a ocorrência vai para revisão manual')
    ap.add_argument('--saida', default=str(AQUI / 'saidas' / 'fotos_testadas'))
    ap.add_argument('--segundos', type=float, default=2.0, help='tempo de cada foto na tela (padrão 2 s)')
    ap.add_argument('--sem-janela', action='store_true', help='não abre janela; só grava os resultados')
    ap.add_argument('--sem-lgpd', action='store_true', help='não borrar cabeças (não recomendado)')
    a = ap.parse_args(argv)

    from ultralytics import YOLO
    pasta, saida = Path(a.pasta), Path(a.saida)
    fotos = sorted(p for p in pasta.iterdir() if p.suffix.lower() in EXT)
    if not fotos:
        sys.exit(f'Nenhuma foto em {pasta}')
    saida.mkdir(parents=True, exist_ok=True)
    gabarito = ler_gabarito(pasta)
    modelo = YOLO(a.modelo)
    coco = AQUI / 'yolo11n.pt'          # modelo COCO original, usado só para achar pessoas e borrar as cabeças
    pessoas = None if a.sem_lgpd else YOLO(str(coco) if coco.exists() else 'yolo11n.pt')
    print(f'{len(fotos)} fotos em {pasta} · modelo {Path(a.modelo).name} · conf {a.conf}')

    # 1) processa todas as fotos (rápido: ~16 ms por foto na GPU) e grava os resultados
    telas, linhas, placar, t0 = [], [], Counter(), time.time()
    for i, p in enumerate(fotos, 1):
        img = imread(p)
        if img is None:
            print('  não consegui ler', p.name); continue
        r = modelo.predict(img, conf=a.conf, verbose=False)[0]
        dets = [(r.names[int(c)], float(s), b) for c, s, b in
                zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
        resp = montar_resposta(dets, a.revisao)
        if pessoas is not None:                   # detecta na foto original; só depois borra para exibir/salvar
            rp = pessoas.predict(img, classes=[0], conf=0.3, verbose=False)[0]
            borrar_cabecas(img, rp.boxes.xyxy.tolist())
        gab = gabarito.get(p.name)
        res = avaliar(resp, gab, a.conf)
        if res:
            placar[res] += 1
        tela = painel(desenhar(img, resp), i, len(fotos), p.name, resp, gab, res, placar, a.segundos)
        imwrite(saida / f'{p.stem}.jpg', tela, quality=88)
        telas.append(tela)
        sug = resp['sugestao'] or {}
        linhas.append({'foto': p.name, 'sugestao': sug.get('nome_pt', ''), 'categoria': sug.get('categoria', ''),
                       'prioridade': sug.get('prioridade', ''), 'confianca': sug.get('confianca', ''),
                       'revisao_manual': resp['revisao_manual'], 'motivo': resp['motivo'],
                       'outras_categorias': '; '.join(sug.get('outras_categorias', [])),
                       'deteccoes': len(resp['deteccoes']),
                       'rotulo_real': gab['nomes_pt'] if gab else '', 'resultado': res or ''})
        print(f"  {i:02d}/{len(fotos)} {p.name:<28} {sug.get('nome_pt', '— nada detectado'):<34} "
              f"{(str(round(sug['confianca'] * 100)) + '%') if sug else '':>5}  {res or ''}")
    seg = time.time() - t0
    with open(saida / 'resultados.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0])); w.writeheader(); w.writerows(linhas)
    resumo = {'fotos': len(linhas), 'segundos': round(seg, 1), **{k: placar[k] for k in COR_RES}}
    sugeriu = placar['categoria certa'] + placar['só categoria errada']
    if gabarito:
        resumo['acerto_quando_sugere'] = round(placar['categoria certa'] / max(1, sugeriu), 3)
    (saida / 'resumo.json').write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nResumo: {resumo}')
    if gabarito:
        print(f"Quando o modelo sugere, acerta {placar['categoria certa']} de {sugeriu} fotos "
              f"({100 * placar['categoria certa'] / max(1, sugeriu):.0f}%).")
    print(f'Fotos com o painel, resultados.csv e resumo.json em {saida}')

    # 2) visualizador
    if a.sem_janela:
        return
    janela = 'BairroVivo Vision - fotos a serem testadas'
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela, LARG, ALT)
    espera = max(1, int(a.segundos * 1000))
    for tela in telas:                           # passa sozinho; Q/Esc ou fechar a janela encerra
        cv2.imshow(janela, tela)
        t_fim = time.time() + espera / 1000
        sair = False
        while time.time() < t_fim:               # espera em fatias curtas para responder rápido ao Q/Esc
            k = cv2.waitKey(50) & 0xFF
            if k in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) < 1:
                sair = True
                break
        if sair:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
