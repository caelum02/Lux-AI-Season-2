- Lux AI 환경의 경우, 유닛의 개수, 팩토리의 개수, 인접 resource의 존재 유무 등에 따라 가능한 Action들이 변함
- Policy를 neural net으로 모델링하기 위해서는 output의 형태(=action space)가 고정되어야한다는 문제가 있음
- 따라서 가능한 모든 valid action을 포괄할 수 있도록 policy network를 모델링하고, invalid action을 배제해나갈 방법을 찾아야 함.
- 이를 해결하기 위해 사용하는 접근으로는 두가지가 있다

| invalid action penalty | invalid action masking |
|---|---|
| invalid action에 대해 negative reward 부가 | invalid action은 배제하고 valid action에 대해서만 샘플링 |
| invalid action의 space가 커질 때 잘 scale x | scalable, 가장 범용적으로 쓰는 것으로 추정됨 |

-  invalid action masking을 사용하는 것이 합리적임.

#### Common Implementation of Invalid action masking
- invalid action에 대해, forward propagation시 invalid action의 logit의 값을 -inf로 설정한다.
![image](https://github.com/caelum02/LuX-AI-Season-2/assets/38996666/85e91876-c73b-415b-b789-e6ba56d70bb7)

#### Questions
- [ ] masking되기 전의 logit은 어떤 값을 모델링한 것인가?
- [x] Policy gradient update는 어떻게 바뀔 것인가?
- [x] Entropy term, KL divergence term에 대한 update는 어떻게 되는가?

#### Policy gradient update는 어떻게 바뀔 것인가?
- Policy gradient theorem (REINFORCE)
```math
 \nabla_\theta J \propto \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum^{T-1}_{t=0} {\nabla_\theta \log{\pi_\theta (A_t|S_t)G_t}}  \right]
```
- policy gradient update (logit perspective, 샘플에서 $A_t=i$일 경우)
```math
\frac{\partial\log{p_i}}{\partial z_j} =
\begin{cases}
 1-p_j & \text{if }i=j\\
 -p_j & \text{if }i\ne j
\end{cases}
```
- 주목할만한 점은 **각 j번째 logit 에 대해 미분계수를 다 더할 경우 0이 된다는 점**이다.
```math
\sum_j\frac{\partial\log{p_i}}{\partial z_j} = 0
```
- 1번 카테고리의 action을 택하여 $G_t=g$로 샘플링되었다고 하면, logit의 업데이트 식은 다음과 같다:
```math
\nabla_{logit}J \propto G_t * (1-p_1, p_2, \dots, p_n ) = 20 * (0.5, -0.1, -0.25, -0.15) (\text{for example})
```


마스킹이 이루어질 경우:
- invalid action은 정합한 환경에서 sampling되지 않으므로 $i=j$ 케이스는 발생하지 않음.
- valid action에 대한 policy update시, $p_{invalid}=0$이므로 invalid action의 logit은 gradient가 0이 된다.

따라서 **마스킹은 invalid action에 대한 policy gradient update를 차단**한다.

- [Huang and Ontanon](https://browse.arxiv.org/pdf/2006.14171.pdf)에 따르면, masking을 고려하지 않고 network를 업데이트할 경우 (즉, policy gradient, entropy 등을 naive하게 계산할 경우) PPO 알고리즘 사용한 경우 target policy와 current policy의 KL divergence가 발산하였음. 

#### Update target에 Entropy term이 더해질 경우?
- $\frac{\partial H(\pi)}{\partial{logit_i}}= -p_i(H(\pi)+\log{p_i})$
- masking할 경우 $p_i=0$이므로 $\nabla_{logit_i}H(\pi)=0$

따라서 **마스킹은 invalid action에 대한 entropy gradient를 차단**한다.

그러므로 참고한 reference https://boring-guy.sh/posts/masking-rl/ 에서는 entropy를 계산할 때 mask에 따라 invalid action의 $plogp$값에 임의로 0을 집어넣는 방식은 합리적이다. action masking은 forward & backward 모두에서 계산 그래프가 끊어지는 것과 동일하기 때문이다.

#### KL divergence 항이 더해질 경우?
- 현재 모델의 output distribution을 $p$, true distribution을 $q$라고 하자.(ex. teacher model)
- $\frac{\partial KL(q||p)}{\partial logit_i}=q_i-p_i$
- invalid action은 p, q에서 둘다 확률이 0이어야 하므로 gradient=0

#### Reference 
1. https://boring-guy.sh/posts/masking-rl/
2. https://browse.arxiv.org/pdf/2006.14171.pdf
3. https://lilianweng.github.io/posts/2018-04-08-policy-gradient/

