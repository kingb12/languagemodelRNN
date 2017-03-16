function reward_maker(target) 
  local function r(sample) 
    local ti = 1
    local lcs = {}
    for i = 0, target:size(1) do
      lcs[i] = {}
      for j = 0, sample:size(1) do
        if i == 0 or j == 0 then
          lcs[i][j] = 0
        else
          lcs[i][j] = -math.huge
        end
      end
    end 

    for i = 1, target:size(1) do
      for j = 1, sample:size(1) do
        if target[i] == target[j] then
          lcs[i][j] = lcs[i-1][j-1] + 1
        else
          lcs[i][j] = math.max(lcs[i-1][j], lcs[i][j-1])
        end
      end
    end

    return lcs[target:size(1)][sample:size(1)]
  end
  return r
end
