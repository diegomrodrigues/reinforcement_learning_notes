![Value function sweeps and final policy for the gambler's problem with ph = 0.4.](./images/image1.png)

The image, labeled as Figure 4.3, illustrates the solution to the Gambler's Problem with ph=0.4. The upper graph depicts value functions obtained through successive sweeps of value iteration, showcasing how the value estimates evolve with capital. Distinct lines represent different sweeps (1, 2, 3, and 32), converging towards the 'Final value function'. The lower graph represents the final policy, showing the stake (bet amount) for each level of capital, revealing the policy's structure and optimal betting strategy.

![Diagrama da iteração da política generalizada (GPI) mostrando o ciclo entre avaliação e melhoria da política.](./images/image2.png)

A imagem representa o diagrama de iteração da política generalizada (GPI), conforme discutido na Seção 4.6 do Capítulo 4. O diagrama ilustra um ciclo iterativo entre a avaliação da política (evaluation), onde a função de valor V é atualizada com base na política π, resultando em Vπ, e a melhoria da política (improvement), onde a política π é aprimorada tornando-se 'greedy' em relação à função de valor V. Este ciclo continua até que a política e a função de valor convirjam para a política ótima π* e a função de valor ótima v*.

![Policy Iteration algorithm: iterative process of policy evaluation and improvement for optimal policy estimation.](./images/image3.png)

The image describes the Policy Iteration algorithm, a dynamic programming approach for estimating an optimal policy \(\pi\) in reinforcement learning, as described in Chapter 4 of the document. It initializes the value function V(s) and policy \(\pi\)(s) arbitrarily, then iteratively performs policy evaluation to improve the value function and policy improvement to refine the policy, as detailed in Section 4.3. The algorithm terminates when the policy stabilizes, returning an approximation of the optimal value function and policy.

![Pseudocode for Value Iteration algorithm, a method for estimating the optimal policy in an MDP.](./images/image4.png)

The image shows the pseudocode for Value Iteration, an algorithm used for estimating an optimal policy in a Markov Decision Process (MDP), as discussed in Chapter 4 of the document. The algorithm initializes a value function V(s) for each state, iteratively updating it based on Bellman's optimality equation until the change in value function, Δ, falls below a threshold θ. This procedure combines policy evaluation and policy improvement steps in each sweep, leading to faster convergence, and outputs a deterministic policy π approximating the optimal policy π∗.

![Pseudocode for Iterative Policy Evaluation, an algorithm for estimating state values under a given policy.](./images/image5.png)

The image presents pseudocode for Iterative Policy Evaluation, an algorithm used to estimate the state-value function V for a given policy π, as described in Section 4.1. The algorithm initializes V(s) arbitrarily for all states s and iteratively updates it using the Bellman equation, stopping when the maximum change in V(s) across all states is below a threshold θ. This process allows for progressively more accurate approximations of the true state values under the specified policy.

![Diagrama representando a interação entre avaliação e melhoria de políticas na iteração da política generalizada (GPI).](./images/image6.png)

A imagem representa o conceito de iteração da política generalizada (GPI) e mostra a interação entre os processos de avaliação e melhoria da política; é mencionada na página 86. O diagrama ilustra que a avaliação da política leva a uma função de valor, que por sua vez é usada para melhorar a política. O objetivo é encontrar uma política e função de valor que sejam ótimas, indicadas por υ* e π*.

![Convergence of iterative policy evaluation on a gridworld, showing improvement from random to optimal policy.](./images/image7.png)

This figure, labeled Figure 4.1, depicts the convergence of iterative policy evaluation on a 4x4 gridworld, as discussed in Chapter 4 of the document. The left column shows the sequence of approximations of the state-value function for a random policy where all actions are equally likely. The right column illustrates the sequence of greedy policies corresponding to the value function estimates, with arrows indicating actions achieving the maximum value, rounded to two significant digits. The figure demonstrates the iterative improvement of policies, progressing from a random policy at k=0 to an optimal policy as k approaches infinity, with policies after the third iteration (k=3) being optimal in this case.

![Policy iteration for Jack's car rental problem, showing policy improvements and the final state-value function.](./images/image8.png)

This figure, labeled as Figure 4.2, illustrates the sequence of policies found by policy iteration on Jack's car rental problem, along with the final state-value function. The first five diagrams represent the policy \(\pi_0\) to \(\pi_4\), indicating the number of cars to be moved between two locations for each possible state (number of cars at each location at the end of the day). Negative numbers indicate transfers from the second to the first location. The figure also depicts the final state-value function \(v_{\pi_4}\) as a 3D surface plot, showing the value for each state (number of cars at each location) after the policy iteration has converged. According to the accompanying text in Chapter 4, each successive policy is a strict improvement over the previous policy, and the last policy is optimal.

![Illustration of a 4x4 gridworld environment with rewards and actions for dynamic programming example.](./images/image9.png)

The image illustrates Example 4.1, a 4x4 gridworld used to demonstrate dynamic programming concepts, as referenced on page 76 of Chapter 4. It displays the grid's structure with numbered states, shaded terminal states, and indicates possible actions (up, down, left, right) with arrows. Additionally, it specifies a reward of -1 for all transitions, representing a cost for each step taken within the environment.
