% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/plot.fmds.R
\name{plot.fmds}
\alias{plot.fmds}
\title{Visualisation of an fmds object}
\usage{
\method{plot}{fmds}(
  x,
  type = c("configuration", "transformation", "fit", "residuals", "shepard", "stress",
    "biplot", "dendrogram", "threshold", "neighbors"),
  markers = NULL,
  labels = NULL,
  ...
)
}
\arguments{
\item{x}{An fmds object .}

\item{type}{type of plot (configuration by default)}

\item{markers}{vector or matrix for pie markers}

\item{labels}{vector with labels}

\item{\dots}{additional arguments to pass}
}
\value{
none
}
\description{
Plot function for a \code{fmds} object. The plot shows the result of \code{fmds}.
}
