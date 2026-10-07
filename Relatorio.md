# Relatório Experimental: SARSA com Aproximação Linear no MountainCar-v0

**Disciplina:** Aprendizado por Reforço  
**Ambiente:** Gymnasium `MountainCar-v0`  
**Algoritmo:** SARSA com Aproximação Linear de Função via *Radial Basis Functions* (RBF)

---

## 1. Setup Experimental

O problema do *MountainCar-v0* consiste em controlar um carro subpotente em um vale unidimensional para alcançar o topo da montanha à direita ($x \ge 0.5$). O espaço de observação contínuo é bidimensional: posição $x \in [-1.2, 0.6]$ e velocidade $\dot{x} \in [-0.07, 0.07]$. As ações disponíveis são discretas: acelerar para a esquerda ($0$), não acelerar ($1$) e acelerar para a direita ($2$).

### 1.1 Modelo e Algoritmo
* **Aproximação Linear:** $Q(s, a) = \boldsymbol{\theta}_a^\top \boldsymbol{\phi}(s)$, onde $\boldsymbol{\theta}_a \in \mathbb{R}^M$ são os pesos da ação $a$ e $\boldsymbol{\phi}(s) \in \mathbb{R}^M$ é o vetor de ativações RBF.
* **Extrator RBF:** Os estados contínuos são normalizados para o intervalo $[0, 1]^2$. São alocados $M = n_{\text{centers}}$ centros Gaussianos distribuídos em uma grade uniforme no espaço normalizado, com $\phi_i(s) = \exp\left(-\frac{\|s_{\text{norm}} - c_i\|^2}{2\sigma^2}\right)$, fixando-se $\sigma = 0.15$.
* **Regra de Atualização (SARSA):** $\boldsymbol{\theta}_a \leftarrow \boldsymbol{\theta}_a + \alpha [r + \gamma Q(s', a') - Q(s, a)] \boldsymbol{\phi}(s)$, com fator de desconto $\gamma = 1.0$.
* **Política:** $\varepsilon$-gulosa com $\varepsilon_{\text{inicial}} = 0.1$, decaimento multiplicativo de $0.995$ por episódio e $\varepsilon_{\text{mín}} = 0.01$.

### 1.2 Protocolo de Testes
Foi explorada uma grade completa de hiperparâmetros variando a taxa de aprendizado $\alpha \in \{0.005, 0.01, 0.02, 0.05, 0.1\}$ e o número de centros RBF $n_{\text{centers}} \in \{9, 16, 25, 36, 49\}$, totalizando 25 combinações. Cada configuração foi avaliada em **10 sementes aleatórias independentes** (seeds 42 a 51) ao longo de até 50.000 passos de ambiente. Registrou-se o número de passos necessários até que o agente atingisse **10 episódios de sucesso** (chegada ao objetivo antes do limite de 200 passos por episódio).

---

## 2. Resultados e Seleção das 5 Configurações

Para análise aprofundada, selecionou-se um conjunto ortogonal de 5 configurações que permite isolar simultaneamente os efeitos da resolução espacial e da taxa de aprendizado:

1. **$\alpha = 0.10$, 49 centros:** Maior agressividade de aprendizado com alta capacidade representacional.
2. **$\alpha = 0.05$, 49 centros:** Configuração de referência (*anchor*), com alta estabilidade e mínima variância.
3. **$\alpha = 0.05$, 25 centros:** Resolução espacial intermediária com taxa de aprendizado ótima.
4. **$\alpha = 0.01$, 49 centros:** Alta resolução espacial, porém com taxa de aprendizado conservadora.
5. **$\alpha = 0.05$, 9 centros:** Resolução espacial grosseira ($3 \times 3$), evidenciando o gargalo de capacidade.

### Tabela 1: Estatísticas de Passos até 10 Sucessos (10 sementes por configuração)

| Configuração | Taxa $\alpha$ | Centros RBF | Mediana | Média $\pm$ Desv. Padrão | Mínimo | Máximo | Taxa Sucesso |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\alpha = 0.10$ (49 centros) | 0.10 | 49 | **3.230** | 3.404,8 $\pm$ 600,6 | 2.749 | 4.612 | **100%** |
| $\alpha = 0.05$ (49 centros) | 0.05 | 49 | 4.530 | 4.489,2 $\pm$ 279,0 | 3.912 | 4.960 | **100%** |
| $\alpha = 0.05$ (25 centros) | 0.05 | 25 | 6.578 | 6.709,2 $\pm$ 413,4 | 6.214 | 7.670 | **100%** |
| $\alpha = 0.01$ (49 centros) | 0.01 | 49 | 11.752 | 11.846,2 $\pm$ 847,0 | 10.450 | 12.957 | **100%** |
| $\alpha = 0.05$ (9 centros) | 0.05 | 9 | 50.000 | 48.106,2 $\pm$ 3.484,6 | 38.542 | 50.000 | **40%** |

---

## 3. Visualização: Boxplot Comparativo

![Boxplot comparativo dos passos até 10 sucessos para as 5 configurações selecionadas.](boxplot.png)

---

## 4. Discussão e Interpretação Teórica

### 4.1 Efeito da Granularidade dos Centros RBF (Capacidade de Representação)
Comparando as configurações com $\alpha = 0.05$ fixo (49 centros, 25 centros e 9 centros):
* **49 centros ($7 \times 7$):** Permite discretizar finamente o espaço de fase (posição $\times$ velocidade). O agente aprende a acumular energia cinética embalando para trás e para frente rapidamente (mediana de 4.530 passos).
* **25 centros ($5 \times 5$):** Apresenta desempenho sólido (mediana de 6.578 passos, aumento de $\approx 45\%$ em relação a 49 centros), demonstrando boa capacidade com menor custo computacional.
* **9 centros ($3 \times 3$):** Provoca sobre-generalização extrema. A distância euclidiana entre centros adjacentes é de $0.5$ no espaço normalizado, tornando impossível distinguir estados próximos com velocidades opostas. O agente sofre com interferência catastrófica na aproximação do valor $Q$, atingindo a meta em apenas 40% das sementes dentro dos 50.000 passos (mediana truncada em 50.000).

### 4.2 Efeito da Taxa de Aprendizado ($\alpha$)
Fixando a malha em 49 centros e variando $\alpha \in \{0.01, 0.05, 0.10\}$:
* **$\alpha = 0.10$:** Proporcionou a convergência mais rápida observada em toda a bateria experimental (mediana de apenas 3.230 passos). Devido à representação esparsa e local das bases RBF, taxas de aprendizado mais elevadas aceleram a propagação do sinal de recompensa negativa/positiva sem instabilizar a função de valor.
* **$\alpha = 0.05$:** Apresentou a **menor dispersão** entre todas as configurações testadas ($\sigma = 279,0$ passos), indicando máxima robustez à semente inicial do ambiente.
* **$\alpha = 0.01$:** Embora alcance 100% de taxa de sucesso, converge de forma significativamente mais lenta (mediana de 11.752 passos, $\approx 3,6\times$ mais demorado que $\alpha = 0.10$). As atualizações incrementais tornam a difusão do valor terminal muito gradual ao longo da trajetória de transições.

---

## 5. Conclusão

Os experimentos confirmam os princípios fundamentais descritos por Sutton & Barto (2018):
1. No MountainCar, uma representação contínua adequada do espaço de estados é pré-requisito indispensável para o sucesso do controle linear; representações de baixa dimensão ($n_{\text{centers}} \le 9$) geram *underfitting* espacial crítico.
2. Com resolução suficiente ($n_{\text{centers}} \ge 25$), o SARSA com RBF demonstra alta robustez estocástica, e taxas de aprendizado moderadas a altas ($\alpha \in [0.05, 0.10]$) otimizam a velocidade de convergência sem incorrer em divergência numérica.

