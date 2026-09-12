# src/config.R -- the R twin of src/config.py.
#
# Same two files, same merge: config.yaml declares every key and is published;
# config.local.yaml supplies the machine paths and is gitignored. See src/config.py for
# why the split exists and what a public checkout sees.
#
#   source("src/config.R")
#   config <- load_config()
#   panel  <- require_path(config, "genetic_tools.pascalx.reference_panel")
#
# LOCALE. yaml::read_yaml() reads through readLines() with the session encoding, so on a
# machine whose declared locale is not actually generated -- R then falls back to C --
# the superscript-2 and multiplication signs in `covariate_names` are mangled and the
# parse dies with a misleading "did not find expected ',' or ']'" pointing at that line.
# Reading with an explicit UTF-8 encoding is what makes this work on a reader's machine,
# and fileEncoding= on read_yaml() is NOT sufficient: it has to be on the read itself.

.mvp_repo_root <- function(start = getwd()) {
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

read_yaml_utf8 <- function(path) {
    yaml::yaml.load(paste(readLines(path, encoding = "UTF-8", warn = FALSE), collapse = "\n"))
}

# overlay wins, recursively -- except that an explicit NULL never overwrites a value, so
# the public file's `null` placeholders cannot blank out what the local file supplies.
deep_merge <- function(base, overlay) {
    if (!is.list(base) || !is.list(overlay)) return(overlay)
    for (key in names(overlay)) {
        value <- overlay[[key]]
        if (is.null(value)) next
        if (is.list(value) && is.list(base[[key]])) {
            base[[key]] <- deep_merge(base[[key]], value)
        } else {
            base[[key]] <- value
        }
    }
    base
}

load_config <- function(path = NULL, local = NULL) {
    root <- if (is.null(path)) .mvp_repo_root() else dirname(normalizePath(path))
    if (is.null(path)) path <- file.path(root, "config.yaml")
    config <- read_yaml_utf8(path)
    if (is.null(local)) {
        from_env <- Sys.getenv("MVP_CONFIG_LOCAL", unset = "")
        local <- if (nzchar(from_env)) from_env else sub("\\.yaml$", ".local.yaml", path)
    }
    if (file.exists(local)) config <- deep_merge(config, read_yaml_utf8(local))
    config
}

config_get <- function(config, dotted, default = NULL) {
    node <- config
    for (part in strsplit(dotted, ".", fixed = TRUE)[[1]]) {
        if (!is.list(node) || is.null(node[[part]])) return(default)
        node <- node[[part]]
    }
    node
}

require_path <- function(config, dotted) {
    value <- config_get(config, dotted)
    if (!is.null(value) && nzchar(value)) return(value)
    stop(dotted, " is not configured.\n",
         "  It is a machine path, so it lives in config.local.yaml (gitignored), not in\n",
         "  config.yaml. On a public checkout that file does not exist and this key has no\n",
         "  value: the script you ran needs cohort-level data rather than the published\n",
         "  deposit. The figure scripts under 05_figures/ fetch their inputs by DOI.\n",
         "  On our machines: add ", dotted, " to config.local.yaml.")
}

# Where figure output belongs: results/main_figures/ or results/supplementary_figures/, each
# flat. The R twin of src/display_item.figure_dir; see it for why.
figures_root <- function(config = NULL) {
    if (is.null(config)) config <- load_config()
    configured <- config_get(config, "output.figures_dir", "results")
    if (startsWith(configured, "/")) configured else file.path(.mvp_repo_root(), configured)
}

figure_dir <- function(section, config = NULL, create = TRUE) {
    sections <- c("main_figures", "supplementary_figures")
    if (!section %in% sections)
        stop("figure_dir('", section, "'): expected one of ", paste(sections, collapse = ", "))
    path <- file.path(figures_root(config), section)
    if (create) dir.create(path, recursive = TRUE, showWarnings = FALSE)
    path
}
