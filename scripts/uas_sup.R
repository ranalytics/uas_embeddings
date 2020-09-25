require(tidyverse)
require(reticulate)
require(Rtsne)
require(plotly)

# Make sure `fasttext` is available to R:
py_module_available("fasttext")

# Load `fasttext`:
ft <- reticulate::import("fasttext")


# ------ Load and prepare the data ------

dat <- read_csv("data/sample_200k.zip")

# Normalise UAS to lower case, add labels, and split data into train and test sets:
set.seed(42)
dat <- dat %>% 
  mutate(uas = tolower(user_agent),
         labeled_uas = paste0("__label__", hardware_type, " ", uas)) %>%
  mutate(dataset = sample(c("train", "test"), n(), replace = TRUE, 
                          prob = c(0.975, 0.025)))

# Write train set into a text file, as expected by fastText:
dat %>% 
  filter(dataset == "train") %>% 
  pull(labeled_uas) %>% 
  writeLines(., "./data/train_data_sup.txt", useBytes = TRUE)

test_data <- dat %>% filter(dataset == "test")


# ------ Supervised training ------

# See https://fasttext.cc/docs/en/python-module.html for details

params <- list(train_path = "./data/train_data_sup.txt",
               lr = 0.05,
               dim = 32L, # vector dimension
               minn = 2L,
               maxn = 6L,
               minCount = 1L,
               minCountLabel = 10L, # min label occurence
               neg = 3L,
               wordNgrams = 2L,
               ws = 3L,
               epoch = 100L,
               loss = 'softmax', # loss function
               thread = 12L)

# Train the model:
m_sup <- ft$train_supervised(input = params$train_path,
                             lr = params$lr, 
                             dim = params$dim, 
                             ws = params$ws, 
                             minCount = params$minCount,
                             minn = params$minn, 
                             maxn = params$maxn, 
                             neg = params$neg, 
                             wordNgrams = params$wordNgrams, 
                             loss = params$loss,
                             epoch = params$epoch, 
                             thread = params$thread)

# Save the model for further use. It can be loaded later as follows:
# m_sup <- ft$load_model(path = "./models/model_sup")
m_sup$save_model(path = "./models/model_sup")


# Calculate embeddings for each UAS from the test set:
emb_sup <- test_data %>% 
  pull(labeled_uas) %>% 
  lapply(., function(x) {
    m_sup$get_sentence_vector(text = x) %>% 
      t() %>% as.data.frame()
  }) %>% 
  bind_rows() %>% 
  setNames(., paste0("f", 1:params$dim))


# Apply tSNE to ease visualisation of the resultant embeddings:
tsne_sup <- Rtsne(emb_sup, 
                  dims = 3, 
                  pca = TRUE,
                  perplexity = 40,
                  verbose = TRUE, 
                  max_iter = 1000,
                  check_duplicates = FALSE)

# Get tSNE coordinates:
tsne_coords_sup <- tsne_sup$Y %>% 
  as.data.frame() %>% 
  setNames(., c("D1", "D2", "D3")) %>%
  as_tibble() %>%
  mutate(label = test_data$hardware_type) %>% 
  filter(label %in% c("computer", "mobile", "server")) # use frequent labels only


# Set figure margins:
m <- list(
  l = 2,
  r = 2,
  b = 2,
  t = 30,
  pad = 1
)


# Visualise results in an interactive plot:
(p <- plot_ly(tsne_coords_sup, 
              x = ~D1,
              y = ~D2,
              z = ~D3,
              color = ~ label, 
              colors = "Set1",
              text = ~ label,
              hoverinfo = "text",
              marker = list(size = 4, opacity = 0.5)) %>%
    add_markers() %>%
    layout(scene = list(xaxis = list(title = 'tSNE dimension 1', 
                                     showspikes = FALSE),
                        yaxis = list(title = 'tSNE dimension 2',
                                     showspikes = FALSE),
                        zaxis = list(title = 'tSNE dimension 3',
                                     showspikes = FALSE)),
           showlegend = TRUE,
           legend = list(title = list(text = '<b> Hardware: </b>')),
           autosize = FALSE, 
           margin = m) %>% 
    config(displayModeBar = FALSE)
)
