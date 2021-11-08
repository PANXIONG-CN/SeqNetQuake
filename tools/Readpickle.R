library(reticulate)
pd <- import("pandas")
pickle_data <- pd$read_pickle("listfile.data")
