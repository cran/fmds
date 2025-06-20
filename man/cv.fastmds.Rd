% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cv.fastmds.R
\name{cv.fastmds}
\alias{cv.fastmds}
\title{Repeated Cross-Validation Penalized Restricted Multidimensional Scaling Function}
\usage{
cv.fastmds(
  delta,
  w = NULL,
  p = 2,
  q = NULL,
  b = NULL,
  lambda = 0,
  alpha = 1,
  grouped = FALSE,
  NFOLDS = 10,
  NREPEATS = 30,
  MAXITER = 1024,
  FCRIT = 1e-08,
  ZCRIT = 1e-06,
  error.check = FALSE,
  echo = FALSE
)
}
\arguments{
\item{delta}{an n by n symmatric and hollow matrix containing dissimilarities.}

\item{w}{an identical sized matrix containing nonnegative weights (all ones when omitted).}

\item{p}{dimensionality (default = 2).}

\item{q}{independent variables (n by h).}

\item{b}{initial regression coefficients (h by p).}

\item{lambda}{regularization penalty parameter(s) (default = 0.0: no penalty).}

\item{alpha}{elastic-net parameter (default = 1.0: lasso only).}

\item{grouped}{boolean for lasso penalty (default = FALSE: ordinary lasso).}

\item{NFOLDS}{number of folds for the k-fold cross-validation.}

\item{NREPEATS}{number of repeats for the repeated k-fold cross-validation.}

\item{MAXITER}{maximum number of iterations (default = 1024).}

\item{FCRIT}{relative convergence criterion function value (default = 0.00000001).}

\item{ZCRIT}{absolute convergence criterion coordinates (default = 0.000001).}

\item{error.check}{extensive check validity input parameters (default = FALSE).}

\item{echo}{print intermediate algorithm results (default = FALSE).}
}
\value{
mserrors mean squared errors for different values of lambda.

stderrors standard errors for mean squared errors.

varnames labels of independent row variables.

coefficients list with final h by p matrices with regression coefficients (lambda order).

lambda sorted regularization penalty parameters.

alpha elastic-net parameter (default = 1.0: lasso only).

grouped boolean for lasso penalty (default = FALSE: ordinary lasso).
}
\description{
\code{cv.fastmds} performs repeated cross-validation for a penalized restricted multidimensional scaling model.
}
\references{
de Leeuw, J., and Heiser, W. J. (1980). Multidimensional scaling with restrictions on the configuration.
            In P.R. Krishnaiah (Ed.), Multivariate analysis (Vol. 5, pp. 501–522).
            Amsterdam, The Netherlands: North-Holland Publishing Company.

            Heiser,W. J. (1987a). Joint ordination of species and sites: The unfolding technique.
            In P. Legendre and L. Legendre (Eds.), Developments in numerical ecology (pp. 189–221).
            Berlin, Heidelberg: Springer-Verlag.

            Busing, F.M.T.A. (2010). Advances in multidimensional unfolding.
            Unpublished doctoral dissertation, Leiden University, Leiden, the Netherlands.
}
