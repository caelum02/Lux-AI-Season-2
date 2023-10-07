Referenced
1. https://browse.arxiv.org/pdf/2006.14171.pdf
2. https://boring-guy.sh/posts/masking-rl/


### invalid action masking
Action space는 현재 State에 따라 달라진다. Lux AI 환경의 경우, 유닛의 개수, 팩토리의 개수, 인접 resource의 존재 유무 등에 따라 가능한 Action들이 변한다. Policy를 neural net으로 모델링하기 위해서는 output의 형태(=action space)가 고정되어야한다는 문제가 있다. 이를 해결하기 위해 neural net으로 모든 가능한 action을 모델링하되 invalid action의 경우 **logit에 masking을 해주는 ** 접근법이 있다. Sampling 시에 

Neural network는 현재 State Action 
logit에 -inf로 masking해 invalid action을 제거하는 접근법
##### invalid action masking은 backpropagation에서 어떤 영향을 주는가?

policy gradient target(REINFORCE)

```math
 \nabla_\theta J=\mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum^{T-1}_{t=0} {\nabla_\theta \log{\pi_\theta (A_t|S_t)G_t}}  \right]
```
```math
\frac{\partial\log{p_i}}{\partial z_j} =
\begin{cases}
 1-p_j & \text{if }i=j\\
 -p_j & \text{if }i\ne j
\end{cases}
```

Policy-gradient 계열에서 위 target은 실제 시뮬레이션을 통한 sampling으로 계산한다. invalid action에 대한 $(A_t, S_t)$의 쌍은 샘플링되지 않으므로, 두번째 식의 $i=j$ 케이스는 발생하지 않음.
valid action에 대해서는, invalid action의 probability가 0이므로 gradient가 0이 된다.

따라서 마스킹은 backpropagation을 차단하는 효과를 발생시킨다.
  
##### Update target에 Entropy term이 더해질 경우?
- $p_i$의 logit $z_i$에 대한 gradient $\frac{\partial H(\pi)}{\partial{z_i}}= -p_i(H(\pi)+\log{p_i})$
- $-\inf$로 masking할 경우 $p_i=0$이므로 **logit에 대한 gradient = 0**
 
### policy는 모든 셀에 대해 계산하되 실제로 유닛이 있는 셀에 대해서만 action을 sampling하는 접근

