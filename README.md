# TSP
一个启发式算法解TSP

## 主要是缝合了遗传算法和粒子群算法，并且在初始化的时候做了两次优化，最后效果还不错。。。。
TSP问题，即Traveling Salesman Problem，旅行商问题。该问题是在寻求单一旅行者由起点出发，通过所有给定的需求点之后，最后再回到原点的最小路径成本。TSP问题的应用领域很广泛，例如物流配送、机器人路径规划等。因此找到一种求解TSP的算法至关重要。但是，要想找到最优解(比如动态规划)，其时间复杂度可以达到惊人的$O(2^n n^2)$，在城市数量较多时，所需的运算时间是极大的。TSP问题已经被证明是一个NP-hard问题，即在P≠NP的假设下，找不到一个多项式时间算法来求解其最优解。随着人工智能的发展，出现了许多独立于问题的独立算法，如蚁群算法、粒子群算法、遗传算法、鱼群算法、狼群算法等等。
  
本文中提出一种新的算法来求解TSP问题，该算法参考了PSO中“粒子”的概念以及GA中“交叉”的概念，在此基础上进行了创新性结合，抛弃了粒子群算法中“速度”的概念，但是保留了某个粒子与自身历史最优解和全局最优解进行信息交换，又不同与遗传算法，这种创新性算法是粒子群算法和遗传算法的结合。经过验证，该算法在求解中国主要城市的TSP问题中性能优异，在对参数进行调优之后，该算法得到实验过程中最短路径（155.15328822999243）的概率高达60%，如图1所示。  
  
![图片](https://user-images.githubusercontent.com/126166790/226162884-d18715ad-e206-4d5b-9ff2-050c221444ca.png)

