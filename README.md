# BairroVivo Vision — detecção de problemas urbanos em fotos de moradores

Componente de **Computer Vision** do **BairroVivo** · Global Solution CitizenTech · FIAP · turma 4ESR · 2026

Um **YOLO11n** treinado por *transfer learning* no dataset público **CIVIC ISSUES** detecta 8 tipos de problema de
zeladoria nas fotos que os moradores enviam (buraco, via danificada, lixo, poste, árvore caída, concreto, pichação e
placa). Cada detecção vira uma **sugestão de ocorrência** — categoria, prioridade, confiança e caixa — que o operador
do BairroVivo confirma ou corrige.

**Vídeo de demonstração:** <https://youtu.be/ROWtHUPAbRE> · **Relatório:** `BairroVivo_Vision_Computer_Vision.pdf`

| Integrante | RM |
|---|---|
| Andre Solér | 98827 |
| Fabrizio Maia | 551869 |
| Rodrigo Paixão | 96869 |
| Victor Asfur | 551684 |
| Vitor Shimizu | 550390 |

---

## 1. Como testar: o app nas fotos de `fotos_a_serem_testadas/`

A pasta `fotos_a_serem_testadas/` tem **56 fotos** (7 de cada classe) do conjunto de **teste** — nenhuma foi vista no
treino. Elas simulam ocorrências enviadas por moradores. O arquivo `gabarito.csv` traz o rótulo real de cada uma,
para o app dizer se o modelo acertou.

**Requisitos:** Python 3.10 ou mais novo. GPU NVIDIA é opcional (sem GPU roda na CPU, ~0,1 s por foto).

```bash
git clone https://github.com/rodrigopaixaosilveira/BairroVivo_ComputerVision.git
cd BairroVivo_ComputerVision
python -m pip install -r requirements.txt
python app_fotos.py
```

Abre uma janela que **passa sozinha pelas 56 fotos, 2 s cada**: à esquerda as caixas detectadas; à direita a
ocorrência que chegaria ao BairroVivo (sugestão, categoria, prioridade, confiança, se vai para revisão manual), o
rótulo real e se acertou, com o placar acumulado. A janela fecha sozinha depois da última foto; **Q** ou **Esc**
encerra antes.

| Opção | Efeito |
|---|---|
| `python app_fotos.py --segundos 3` | muda o tempo de cada foto na tela (padrão 2 s) |
| `python app_fotos.py --sem-janela` | só processa e grava os resultados |
| `python app_fotos.py --pasta minhas_fotos` | roda em outra pasta com qualquer `.jpg`/`.png` (o gabarito é opcional) |
| `--conf 0.35` · `--revisao 0.50` | confiança mínima de uma detecção · abaixo disso vai para revisão manual |

Os resultados ficam em `saidas/fotos_testadas/`: cada foto com o painel, `resultados.csv` (uma linha por foto) e
`resumo.json`. Cabeças de pessoas são borradas antes de exibir e salvar (LGPD), usando o `yolo11n.pt` original (COCO)
só para achar pessoas.

> Para usar a GPU no Windows, instale o PyTorch com CUDA antes do passo do `pip`
> (`pip install torch --index-url https://download.pytorch.org/whl/cu124`). O app escolhe a GPU sozinho quando existe.

### Resultado nas 56 fotos

| Resultado por foto | Quantas | O que acontece no BairroVivo |
|---|---|---|
| sugeriu a categoria certa | **44** (79%) | ocorrência pré-classificada; o operador só confirma |
| não detectou nada | 11 (20%) | vai para **revisão manual** (o modelo não chuta) |
| sugeriu só categoria errada | 1 (2%) | o operador corrige |

**Quando o modelo sugere algo, acerta a categoria em 44 de 45 fotos (98%).** O ponto fraco é deixar passar (recall),
não errar a categoria. As 56 fotos foram processadas em ~7 s numa RTX 3050 (incluindo carregar o modelo).

---

## 2. Por que visão computacional aqui

| Sem o modelo | Com o modelo |
|---|---|
| o morador escolhe a categoria (e erra com frequência) | a categoria é sugerida pela imagem |
| o operador abre e tria cada foto à mão | as ocorrências chegam ordenadas por prioridade |
| vários problemas na mesma foto passam despercebidos | cada problema detectado vira uma sugestão separada |

## 3. Dados

**CIVIC ISSUES v3** — Roboflow Universe, autor *Deshna*, licença **CC BY 4.0**
(<https://universe.roboflow.com/deshna/civic-issues-lroee>), formato YOLOv11.

| | Original |
|---|---|
| Imagens | 43.584 (train 34.980 · valid 4.815 · test 3.787) |
| Fotos de origem distintas | 23.004 (o Roboflow gera ~2 cópias aumentadas de cada foto) |
| Classes | 9 (*Dead_Animal_Pollution* saiu por ter só 192 imagens) |

**Vazamento encontrado e corrigido.** As cópias aumentadas de uma foto têm nomes como `foto_jpg.rf.<hash>.jpg`;
agrupando por esse nome, **1.877 fotos aparecem em mais de um split** do original, e o split original deixava classes
inteiras fora do teste. O notebook agrupa as cópias por foto de origem, usa **uma imagem por grupo** e sorteia 80/10/10
**por grupo** — nenhuma foto do teste (nem cópia aumentada dela) esteve no treino.

Para caber no tempo de treino de uma RTX 3050, foi usada uma amostra com **cota por classe**:
treino **6.081** imagens · validação **908** · teste **911**.

| Classe do dataset | Nome no app | Categoria BairroVivo | Prioridade |
|---|---|---|---|
| pothole | Buraco | Buraco / pavimento | alta |
| road_crack | Via danificada | Buraco / pavimento | média |
| Garbage | Lixo / descarte irregular | Descarte irregular | média |
| Damaged_Electric_Poles | Poste ou fiação danificada | Iluminação / rede elétrica | alta |
| Fallen_Trees | Árvore caída | Via obstruída | alta |
| Damaged_Concrete_Structures | Estrutura de concreto danificada | Calçada / acessibilidade | média |
| Graffiti | Pichação / vandalismo | Mobiliário urbano | baixa |
| Damaged_Road_Signs | Placa danificada | Mobiliário urbano | média |

O dataset (~400 MB) não está no repositório; para retreinar, baixe o export do Roboflow e rode o notebook.

## 4. Modelo e treino

* **YOLO11n** (Ultralytics), detecção de objetos, 2,6 M de parâmetros — escolhido por rodar em **tempo real**.
* **Transfer learning**: parte dos pesos treinados no COCO; a cabeça de classes é trocada pelas 8 classes acima e a
  rede inteira é treinada no nosso dataset.
* 40 épocas, imagem 640 px, batch 16, aumentos padrão (mosaico, espelhamento, cor, escala), RTX 3050 Laptop 6 GB,
  ~6,5 h.

| Época | mAP50 (validação) | Precisão | Recall |
|---|---|---|---|
| 1 | 0,21 | 0,36 | 0,27 |
| 10 | 0,44 | 0,52 | 0,42 |
| 20 | 0,51 | 0,57 | 0,48 |
| 30 | 0,55 | 0,63 | 0,52 |
| 40 | **0,59** | 0,64 | 0,55 |

O erro de validação caiu até o fim (sem *overfitting*); a curva achatou nas últimas 10 épocas.

## 5. Resultados no teste (911 imagens nunca vistas)

| Métrica | Valor |
|---|---|
| **mAP50** | **0,578** |
| mAP50-95 | 0,352 |
| Precisão | 0,621 |
| Recall | 0,539 |
| Velocidade | **63 FPS** na RTX 3050 (≈16 ms por imagem) |

| Classe | Precisão | Recall | F1 | mAP50 |
|---|---|---|---|---|
| Placa danificada | 0,89 | 0,89 | 0,89 | 0,96 |
| Árvore caída | 0,70 | 0,67 | 0,69 | 0,75 |
| Pichação | 0,55 | 0,65 | 0,60 | 0,62 |
| Poste / fiação danificada | 0,59 | 0,57 | 0,58 | 0,61 |
| Lixo / descarte irregular | 0,65 | 0,48 | 0,55 | 0,56 |
| Buraco | 0,61 | 0,46 | 0,52 | 0,48 |
| Trinca na via | 0,53 | 0,42 | 0,47 | 0,43 |
| Concreto danificado | 0,45 | 0,17 | 0,25 | 0,21 |

**mAP50** = qualidade média da detecção (acerto quando a caixa cobre ≥ 50% da real e a classe está certa).
**Precisão** = das detecções feitas, quantas estavam certas. **Recall** = dos problemas existentes, quantos foram
achados. Curvas e matriz de confusão em `runs/bairrovivo_vision_teste/`.

* Objetos com forma clara (placa, árvore, poste, pichação) vão bem.
* **Concreto** e **trincas** são os mais fracos: não têm contorno definido e o tamanho das caixas varia muito entre
  anotadores.
* Recall 0,54: o modelo **não substitui** o operador, ele **sugere** — por isso a confiança baixa gera
  `revisao_manual`.

## 6. Integração com o BairroVivo

`analisar_ocorrencia.py` → `analisar_imagem(foto, modelo)` é o que o backend chama quando chega uma foto:

```json
{
  "sugestao": {"categoria": "Via obstruída", "nome_pt": "Árvore caída", "prioridade": "alta",
               "confianca": 0.87, "outras_categorias": []},
  "deteccoes": [{"classe": "Fallen_Trees", "categoria": "Via obstruída", "prioridade": "alta",
                 "confianca": 0.87, "caixa_xyxy": [12.0, 4.5, 610.3, 636.0]}],
  "revisao_manual": false,
  "motivo": "ok"
}
```

* Confiança < 0,50 ou nada detectado → `revisao_manual: true`.
* `outras_categorias`: outros problemas na mesma foto (ex.: lixo **e** poste caído).

Linha de comando: `python analisar_ocorrencia.py foto.jpg --modelo modelos/bairrovivo_vision.pt --salvar saida.jpg`

## 7. Demo em vídeo, em tempo real (`app_vision.py`)

O mesmo modelo roda em vídeo com rastreamento (**ByteTrack**): cada objeto que aparece em pelo menos 5 frames vira
**uma** ocorrência, com momento, prioridade e um recorte como foto-evidência. O vídeo `videos/demo_fotos_moradores.mp4`
junta as mesmas 56 fotos, 2 s cada.

```bash
python app_vision.py --video videos/demo_fotos_moradores.mp4 --modelo modelos/bairrovivo_vision.pt --salvar
```

Teclas: espaço pausa · Q/Esc fecha. Opções: `--conf`, `--min-frames`, `--inicio/--fim`, `--sem-janela`. No Windows o
vídeo precisa estar num caminho sem acento (limitação do OpenCV).

No vídeo, a mesma seleção dá 35 categorias certas, 20 sem detecção e 1 errada: cada foto vira frames reduzidos para
caber na tela e só vira ocorrência depois de 5 frames seguidos no rastreador, o que é mais exigente que analisar a foto
original como faz o `app_fotos.py`.

## 8. Reproduzir o pipeline

`01_bairrovivo_vision.ipynb`, seção por seção: análise do dataset (2), vazamento (3), dataset final (4), treino (5),
avaliação no teste (6), integração (7), demo em vídeo (8) e app de fotos (9). O caminho do export do Roboflow fica na
variável `ORIGEM` da seção 1. Retreinar leva ~6,5 h numa RTX 3050; para só avaliar, pule a seção 5.

## 9. Arquivos

| Caminho | O que é |
|---|---|
| `app_fotos.py` | **app de teste**: roda o modelo nas fotos de `fotos_a_serem_testadas/` |
| `fotos_a_serem_testadas/` | 56 fotos do teste + `gabarito.csv` (rótulo real de cada uma) |
| `modelos/bairrovivo_vision.pt` · `metricas_vision.json` | modelo treinado e métricas do teste |
| `analisar_ocorrencia.py` | integração: foto → JSON da ocorrência |
| `app_vision.py` | app em vídeo, em tempo real, com painel e rastreamento |
| `vision_lib.py` | leitura do dataset, vazamento, split por origem, cota por classe, mapeamento para o BairroVivo |
| `montar_demo_fotos.py` | monta o vídeo da demo a partir das fotos do teste |
| `01_bairrovivo_vision.ipynb` | pipeline completo |
| `yolo11n.pt` | YOLO11n original (COCO): ponto de partida do treino e detector de pessoas para o borrão LGPD |
| `analise/` | estatísticas do dataset, relatório de vazamento, mosaicos por classe, exemplos da integração |
| `runs/` | curvas de treino, matriz de confusão e métricas do teste |
| `videos/` | vídeo de demonstração (`BairroVivo_Vision_demo.mp4`), vídeo de entrada do `app_vision.py` e a lista das fotos usadas |
| `BairroVivo_Vision_Computer_Vision.pdf` | relatório do trabalho |

## 10. Limitações e próximos passos

* Dataset de fora do Brasil (Índia, Indonésia, China, EUA…): piso, lixo e postes daqui podem ser diferentes. Próximo
  passo: rotular algumas centenas de fotos reais do BairroVivo e fazer *fine-tuning*.
* Classes amorfas (trinca, concreto) pedem rótulos mais consistentes ou segmentação em vez de caixa.
* Modelo maior (YOLO11s) e o dataset inteiro devem subir o mAP, ao custo de mais horas de treino.
* O modelo foi feito para fotos enquadradas no problema; para varrer vídeo contínuo de rua precisaria de imagens de
  rua "normal" no treino (árvores em pé, fachadas e letreiros geram alarmes falsos).

## Licença dos dados

Fotos e rótulos: CIVIC ISSUES v3, Deshna / Roboflow Universe, **CC BY 4.0**. As fotos de `fotos_a_serem_testadas/`
são do conjunto de teste desse dataset (de outros países), usadas para simular o envio de moradores.
