# Fetch deposited figure data by DOI, from R.
#
# Author: Michael Beyeler (github.com/mjbeyeler)
#
# Thin wrapper over src/fetch_data.py -- deliberately not a second implementation.
# DOI resolution, caching and checksum verification are subtle enough that two
# copies would drift; this shells out to the Python one and parses its
# machine-readable --paths output.
#
#   source("05_figures/lib/fetch_data.R")
#   paths <- fetch_data("figure_intermediates")
#   df <- read.csv(paths[["Fig3_01_a_snp_manhattan_dtif_thinned.csv"]])
#
# Files already cached are not re-downloaded, so repeat runs work offline.

# The Python interpreter that has this project's dependencies. Figure scripts shell out
# to Python in two places -- fetching the deposit, and drawing the feature-set Venn from
# its deposited counts (05_figures/lib/render_venn3.py) -- and both must agree on which
# interpreter that is, so readers only ever have one thing to set.
#
# The pixi environment is preferred over whatever `python` happens to be on PATH, because
# R runs in its own isolated pixi environment (`-e r`, no-default-feature) whose python
# deliberately does NOT carry matplotlib: a second matplotlib there could render the Venn
# at a different version than the one the results were produced with. So `pixi run -e r
# Rscript ...` sees a python that cannot draw panel c unless we point at the default
# environment explicitly. A reader without pixi falls through to `python` and can override
# with MVP_PYTHON.
mvp_python <- function(repo_root = NULL) {
    from_env <- Sys.getenv("MVP_PYTHON", unset = "")
    if (nzchar(from_env)) return(from_env)

    if (is.null(repo_root)) repo_root <- find_repo_root()
    env_dir <- file.path(repo_root, ".pixi", "envs", "default")
    # conda lays a Windows environment out without bin/: python.exe sits at its root.
    pixi_python <- if (.Platform$OS.type == "windows") {
        file.path(env_dir, "python.exe")
    } else {
        file.path(env_dir, "bin", "python")
    }
    if (file.exists(pixi_python)) return(pixi_python)

    "python"
}

fetch_data <- function(source,
                       files = NULL,
                       repo_root = NULL,
                       python = NULL) {
    if (is.null(repo_root)) repo_root <- find_repo_root()

    python <- if (!is.null(python)) python else mvp_python(repo_root)

    args <- c("-m", "src.fetch_data", source, "--paths")
    if (!is.null(files)) args <- c(args, as.vector(rbind("--file", files)))

    old <- setwd(repo_root)
    on.exit(setwd(old), add = TRUE)

    # stderr is left attached so download progress reaches the console; only stdout
    # is captured, and --paths guarantees stdout is just the path table.
    output <- suppressWarnings(system2(python, args, stdout = TRUE, stderr = ""))
    status <- attr(output, "status")

    if (!is.null(status) && status != 0) {
        stop(sprintf(paste0(
            "fetching '%s' failed (exit %d).\n",
            "  Check that the deposit is published and config.yaml has its concept DOI:\n",
            "    %s -m src.fetch_data\n",
            "  Set MVP_PYTHON if '%s' is not the interpreter with this project's deps."),
            source, status, python, python))
    }
    if (length(output) == 0) {
        stop(sprintf("fetching '%s' returned no files", source))
    }

    parts <- strsplit(output, "\t", fixed = TRUE)
    bad <- lengths(parts) != 2
    if (any(bad)) {
        stop(sprintf("unparseable output from fetch_data.py:\n  %s",
                     paste(output[bad], collapse = "\n  ")))
    }
    paths <- setNames(vapply(parts, `[`, character(1), 2),
                      vapply(parts, `[`, character(1), 1))

    missing <- paths[!file.exists(paths)]
    if (length(missing)) {
        stop(sprintf("fetch reported files that are not on disk:\n  %s",
                     paste(missing, collapse = "\n  ")))
    }
    paths
}

# Locate the repository root by walking up from the script (or working) directory
# until config.yaml is found, so figure scripts run from anywhere.
find_repo_root <- function(start = NULL) {
    if (is.null(start)) {
        args <- commandArgs(trailingOnly = FALSE)
        file_arg <- grep("^--file=", args, value = TRUE)
        start <- if (length(file_arg)) {
            dirname(normalizePath(sub("^--file=", "", file_arg[1])))
        } else {
            getwd()
        }
    }
    dir <- normalizePath(start, mustWork = FALSE)
    repeat {
        if (file.exists(file.path(dir, "config.yaml"))) return(dir)
        parent <- dirname(dir)
        if (parent == dir) {
            stop("could not locate the repository root (no config.yaml found above ", start, ")")
        }
        dir <- parent
    }
}

# Convenience: fetch one file and return its path.
fetch_data_file <- function(source, name, ...) {
    unname(fetch_data(source, files = name, ...)[[name]])
}

# Read config.yaml regardless of the machine's locale.
#
# yaml::read_yaml() reads through readLines() with the session encoding, so on a
# machine whose declared locale is not actually generated (R then falls back to C)
# the non-ASCII characters in config.yaml -- the superscript-2 and multiplication
# signs in covariate_names -- are mangled and the parse fails with a misleading
# "did not find expected ',' or ']'". Reading with an explicit UTF-8 encoding makes
# figure scripts work on any reader's machine. Note fileEncoding= on read_yaml() is
# NOT sufficient; the encoding has to be set on the read itself.
# One config reader for the whole repository: src/config.R merges config.local.yaml over
# config.yaml (machine paths) and carries the UTF-8 read described above. Sourced at TOP
# LEVEL, not inside read_config(), so that a script sourcing this file also gets
# load_config() and figure_dir() -- with local = TRUE they would live and die inside the
# function body, and every caller would get "could not find function figure_dir".
source(file.path(find_repo_root(), "src", "config.R"))

read_config <- function(path = NULL) {
    if (is.null(path)) path <- file.path(find_repo_root(), "config.yaml")
    load_config(path)
}
