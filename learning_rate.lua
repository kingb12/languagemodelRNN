function learning_rate(lr, p1, diff, avg_diff)
  return lr * p1 * (diff / avg_diff)
end
