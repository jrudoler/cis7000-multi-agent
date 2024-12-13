library(tidyverse)
library(lme4)
library(lmerTest)
library(stargazer)

# Load the data
data <- read_csv("data/grid_results_temp_0.1.csv")

# add a column indicating if the majority vote matches the true label
data <- data %>%
  mutate(correct = ifelse(vote_majority == label, 1, 0))

# calculate the by grouping of n_agents and m_examples
grid_data <- data %>%
  group_by(n_agents, m_examples) %>%
  summarise(accuracy = mean(correct))

grid_data %>%
  ggplot(aes(x = m_examples, y = accuracy)) +
  geom_point(aes(color = factor(n_agents))) +
  geom_smooth(method = "lm") +
  scale_x_log10() +
  # scale_y_log10() +
  labs(title = "Accuracy vs. Number of Examples",
       x = "Log(Number of Examples)",
       y = "Accuracy",
       color = "Number of Agents")
# save the plot
ggsave("figures/accuracy_vs_examples.png")

# fit a linear model
# model <- lm(correct ~ log10(m_examples)*log10(n_agents), data = data)
# summary(model)

model <- lm(log10(accuracy) ~ log10(m_examples)*log10(n_agents), data = grid_data)
summary(model)

# export model summary as a tex table
stargazer(model, type = "latex", out = "tables/model_summary.tex", label = "tab:model_summary")

# fit a mixed effects model
me_model <- lmer(correct ~ log10(m_examples) + (1 | n_agents), data = data)
summary(me_model)
anova(me_model)

# export model summary as a tex table
class(me_model) <- "lmerMod"
stargazer(me_model, type = "latex", out = "tables/me_model_summary.tex")
