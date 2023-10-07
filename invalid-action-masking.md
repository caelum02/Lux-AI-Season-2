Referenced
1. https://browse.arxiv.org/pdf/2006.14171.pdf
2. https://boring-guy.sh/posts/masking-rl/


### logit에 -inf로 masking해 invalid action을 제거하는 접근법
policy gradient target(REINFORCE)

```math
 \nabla_\theta J=\mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum^{T-1}_{t=0} {\nabla_\theta \log{\pi_\theta (A_t|S_t)G_t}}  \right]
```


- Policy-gradient 계열에서 위 target은 Off-polciy/On-policy sampling을 통해 계산하는데 invalid action에 대한 $(A_t, S_t)$의 쌍은
샘플링될 가능성이 없음
### Update target에 Entropy term이 더해질 경우?
- $p_i$의 logit $z_i$에 대한 gradient $\frac{\partial H(\pi)}{\partial{z_i}}= -p_i(H(\pi)+\log{p_i})$
- $-\inf$로 masking할 경우 $p_i=0$이므로 **logit에 대한 gradient = 0**
 
### policy는 모든 셀에 대해 계산하되 실제로 유닛이 있는 셀에 대해서만 action을 sampling하는 접근

