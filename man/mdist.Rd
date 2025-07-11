% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/mdist.R
\name{mdist}
\alias{mdist}
\title{Mixed Measurement Level Euclidean Distances Function}
\usage{
mdist(
  data,
  level = rep("numeric", ncol(data)),
  scale = FALSE,
  error.check = FALSE
)
}
\arguments{
\item{data}{an n (objects) by m (variables) numerical data matrix .}

\item{level}{measurement level variables: 1:numerical, 2:ordinal, 3:nominal (default = 1).}

\item{scale}{boolean specifying scaling of distances such that sum-of-squares are n times n.}

\item{error.check}{extensive check validity input parameters (default = FALSE).}
}
\value{
'dist' object with Euclidean distances between objects.
}
\description{
\code{fastmixed} returns Euclidean distances for variables from mixed measurement levels.
}
\references{
Busing (2025).
            A Consistent Distance Measure for Mixed Data:
            Bridging the Gap between Euclidean and Chi-Squared Distances.
            Manuscript in progress.
}
\author{
Frank M.T.A. Busing
}
