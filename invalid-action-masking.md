Referenced
1. https://browse.arxiv.org/pdf/2006.14171.pdf
2. https://boring-guy.sh/posts/masking-rl/


### logit에 -inf로 masking해 invalid action을 제거하는 접근법
policy gradient target $J(\theta)= E\left[ Q^{\pi}(S_t, A_t) \nabla_\theta \log{\pi(A_t|S_t)} \right]$
- Policy-gradient 계열에서 위 target은 Off-polciy/On-policy sampling을 통해 계산하는데 invalid action에 대한 $(A_t, S_t)$의 쌍은
샘플링될 가능성이 없음
- Exploration을 촉진하기 위한 Entropy term이 더해지는 경우, invalid action이 계산 그래프에 아예 포함되어서는 안됨. Entropy를 계산할 때 아예 0을 넣어버리는 것이 바람직하다고 보여짐.
- Invalid action의 logit에 대해 아무런 업데이트가 이루어지지 않는 것이 바람직한가? 혹은 Invalid action의 logith -inf에 가까워지도록 저해하는 업데이트가 바람직한가?
 
### grid의 모든 cell에 대해 policy를 계산한 뒤, 
