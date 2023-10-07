Referenced
1. https://boring-guy.sh/posts/masking-rl/
2. https://browse.arxiv.org/pdf/2006.14171.pdf
3. https://lilianweng.github.io/posts/2018-04-08-policy-gradient/

- Lux AI 환경의 경우, 유닛의 개수, 팩토리의 개수, 인접 resource의 존재 유무 등에 따라 가능한 Action들이 변함
- Policy를 neural net으로 모델링하기 위해서는 output의 형태(=action space)가 고정되어야한다는 문제가 있음
- 따라서 가능한 모든 valid action을 포괄할 수 있도록 policy network를 모델링하고, invalid action을 배제해나갈 방법을 찾아야 함.
- 이를 해결하기 위해 사용하는 접근으로는 두가지가 있다

| invalid action penalty | invalid action masking |
|---|---|
| invalid action에 대해 negative reward 부가 | invalid action은 배제하고 valid action에 대해서만 샘플링 |
| invalid action의 space가 커질 때 잘 scale x | scalable, 가장 범용적으로 쓰는 것으로 추정됨 |

-  IAM(invalid action masking)을 사용하는 것이 합리적으로 보임

#### Questions
- 모델이 추정한 모든 valid/invalid action의 logit 중 일부만을 택하고 일부는 택하지 않는 것은 모델의 관점에서 어떤 함의를 갖는가? policy network는 어떤 함수를 모델링하는 것인가?
- 모델은 그렇다면 어떻게 학습시켜야하나? 
- 업데이트 식에 Entropy term, KL divergence term (Ex. SAC)에 대한 gradient가 추가될 때 mask된 logit의 gradient는 발산하지 않는가?

#### Common Implementation of Invalid action masking
- invalid action에 대해, forward propagation시 invalid action의 logit의 값을 -inf로 설정한다.
![image](https://github.com/caelum02/LuX-AI-Season-2/assets/38996666/85e91876-c73b-415b-b789-e6ba56d70bb7)

#### invalid action masking은 backpropagation에서 어떤 영향을 주는가?
- Policy gradient target (REINFORCE)
```math
 \nabla_\theta J=\mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum^{T-1}_{t=0} {\nabla_\theta \log{\pi_\theta (A_t|S_t)G_t}}  \right]
```
- policy gradient update (logit perspective)
```math
\frac{\partial\log{p_i}}{\partial z_j} =
\begin{cases}
 1-p_j & \text{if }i=j\\
 -p_j & \text{if }i\ne j
\end{cases}
```

- Policy-gradient 계열에서 위 target은 실제 시뮬레이션을 통한 sampling으로 계산됨
- invalid action에 대한 $(A_t, S_t)$의 쌍은 샘플링되지 않으므로, 두번째 식의 $i=j$ 케이스는 발생하지 않음.
- valid action에 대해서는, invalid action의 probability가 0이므로 gradient가 0이 된다.

따라서 **마스킹은 policy gradient의 backpropagation을 차단하는 효과**를 발생시킨다.

###### Update target에 Entropy term이 더해질 경우?
- $p_i$의 logit $z_i$에 대한 gradient $\frac{\partial H(\pi)}{\partial{z_i}}= -p_i(H(\pi)+\log{p_i})$
- $-\inf$로 masking할 경우 $p_i=0$이므로 **logit에 대한 gradient = 0**

그러므로 참고한 reference https://boring-guy.sh/posts/masking-rl/ 에서는 entropy를 계산할 때 mask에 따라 invalid action의 $plogp$값에 임의로 0을 집어넣는 방식은 합리적이다. 

###### KL divergence 항이 더해질 경우?
 
### policy는 모든 셀에 대해 계산하되 실제로 유닛이 있는 셀에 대해서만 action을 sampling하는 접근

