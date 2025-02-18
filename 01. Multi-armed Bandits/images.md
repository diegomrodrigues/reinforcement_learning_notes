![Comparativo do desempenho do algoritmo gradient bandit com e sem baseline de recompensa no teste de 10 braços.](./images/image1.png)

A imagem, identificada como Figura 2.5 no documento, apresenta um gráfico comparativo do desempenho médio do algoritmo gradient bandit com e sem uma linha de base de recompensa. Os resultados são visualizados para um cenário de teste de 10 braços, no qual os q*(a) são configurados para estarem próximos de +4 em vez de zero, conforme mencionado no texto adjacente. As curvas representam a porcentagem de ações ótimas tomadas ao longo de 1000 passos, com distintas linhas representando o algoritmo com e sem a baseline, cada qual com taxas de aprendizado (α) de 0.1 e 0.4. Esta figura demonstra o impacto da baseline e da taxa de aprendizado no desempenho do algoritmo em um ambiente de aprendizado por reforço.

![Parameter study comparing bandit algorithms, showing average reward over 1000 steps as a function of algorithm-specific parameters.](./images/image2.png)

This image, identified as Figure 2.6 in Chapter 2 of the document, presents a parameter study comparing various bandit algorithms. The graph displays the average reward obtained over the first 1000 steps as a function of each algorithm's key parameter, showcasing characteristic inverted-U shapes indicating optimal performance at intermediate parameter values for epsilon-greedy, UCB, gradient bandit, and optimistic initialization methods; the x-axis represents parameter values (epsilon, alpha, c, Q0), and the y-axis represents the average reward.

![Comparative performance of optimistic vs. realistic action-value initialization in a 10-armed testbed, illustrating the impact on optimal action selection over time.](./images/image3.png)

The image, labeled as 'Figure 2.3' (page 34), presents a comparative analysis of two action-value methods on a 10-armed testbed, illustrating the effect of optimistic initial action-value estimates. The graph plots '% Optimal action' against 'Steps' for two strategies: 'Optimistic, greedy' (Q1=5, ε=0) and 'Realistic, ε-greedy' (Q1=0, ε=0.1), both utilizing a constant step-size parameter α = 0.1. The curves demonstrate how an optimistic initial value leads to initial underperformance due to increased exploration but eventually surpasses the realistic approach as exploration decreases. This highlights the trade-off between initial exploration and long-term performance in bandit problems, emphasizing the role of initial estimates in guiding action selection.

![Pseudocódigo de um algoritmo de bandit simples com estratégia ε-greedy para exploração e explotação.](./images/image4.png)

A imagem apresenta o pseudocódigo de um algoritmo simples de bandit, conforme descrito no Capítulo 2 do documento. O algoritmo inicializa estimativas de valor (Q(a)) e contadores (N(a)) para cada ação 'a' de 1 a 'k'. Em um loop contínuo, ele seleciona uma ação 'A' com base em uma estratégia ε-greedy, executa essa ação usando a função 'bandit(A)', atualiza o contador N(A) e atualiza a estimativa de valor Q(A) usando a recompensa 'R' recebida. Este algoritmo é fundamental para entender os métodos de aprendizado por reforço explorados no capítulo.

![Distribuições de recompensa para um problema de bandit de 10 braços.](./images/image5.png)

A imagem, identificada como Figura 2.1 no documento, ilustra um exemplo de problema de bandit do testbed de 10 braços, conforme mencionado na Seção 2.3. Ela representa a distribuição de recompensas para cada uma das dez ações, onde o valor verdadeiro q*(a) de cada ação é amostrado de uma distribuição normal com média zero e variância unitária, e as recompensas reais são amostradas de uma distribuição normal com média q*(a) e variância unitária, como indicado pelas distribuições em cinza.

![Average performance of ε-greedy action-value methods on a 10-armed testbed, demonstrating the exploration-exploitation trade-off.](./images/image6.png)

The image, labeled as Figure 2.2 in the document, illustrates the average performance of ε-greedy action-value methods on the 10-armed testbed, data averaged over 2000 runs. The figure consists of two subplots: the upper plot shows the 'Average reward' over 'Steps' for different ε values (0, 0.01, and 0.1), and the lower plot displays the '% Optimal action' selected over 'Steps' for the same ε values. The plots serve to visually compare and contrast the performance of the greedy method (ε=0) with the ε-greedy methods (ε=0.01 and ε=0.1), highlighting the trade-off between exploration and exploitation in the context of the k-armed bandit problem as discussed in Section 2.3.

![Average performance comparison of UCB and ε-greedy action selection on a 10-armed testbed.](./images/image7.png)

This figure, Figure 2.4 from Chapter 2 of the document, illustrates the average performance of UCB action selection compared to ε-greedy action selection on a 10-armed testbed. The x-axis represents the number of steps taken (up to 1000), while the y-axis shows the average reward obtained. The UCB algorithm (with c=2) generally outperforms the ε-greedy algorithm (with ε=0.1) after an initial phase where the UCB selects randomly among the as-yet-untried actions.
