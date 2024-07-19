# Download the data from the Dryad repository
# https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq

# Prepare the data directory, this is by default located in the examples/speechbci directory
mkdir -p data && cd data

# Download the data (essential)
wget -O "README.md" https://datadryad.org/stash/downloads/file_stream/2547377
wget -O "competitionData_readme.txt" https://datadryad.org/stash/downloads/file_stream/2547362
wget -O "competitionData.tar.gz" https://datadryad.org/stash/downloads/file_stream/2547369
tar -xzf "competitionData.tar.gz"
wget -O "languageModel_readme.txt" https://datadryad.org/stash/downloads/file_stream/2547360
wget -O "languageModel.tar.gz" https://datadryad.org/stash/downloads/file_stream/2547356
tar -xzf "languageModel.tar.gz"
wget -O "languageModel_5gram.tar.gz" https://datadryad.org/stash/downloads/file_stream/2547359
tar -xzf "languageModel_5gram.tar.gz"

# Download the full data (optional)
# wget -O "derived_readme.txt" https://datadryad.org/stash/downloads/file_stream/2547357
# wget -O "derived.tar.gz" https://datadryad.org/stash/downloads/file_stream/2547370 | tar xz
# wget -O "diagnosticBlocks_readme.txt" https://datadryad.org/stash/downloads/file_stream/2547358
# wget -O "diagnosticBlocks.tar.gz" https://datadryad.org/stash/downloads/file_stream/2547371 | tar xz
# wget -O "sentences_readme.txt" https://datadryad.org/stash/downloads/file_stream/2547361
# wget -O "sentences.tar.gz" https://datadryad.org/stash/downloads/file_stream/2547372 | tar xz
# wget -O "tuningTasks_readme.txt" https://datadryad.org/stash/downloads/file_stream/2547363
# wget -O "tuningTasks.tar.gz" https://datadryad.org/stash/downloads/file_stream/2547373 | tar xz
