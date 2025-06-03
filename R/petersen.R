#' Mark-Recapture Population Size Estimator
#'
#' \code{petersen} returns the estimated population size based on two independent equally-sized samples
#'
#' @param first vector with first sample identifiers (local minima)
#' @param second vector with second sample identifiers (local minima)
#'
#' @return population size estimate
#'
#' @references Busing (2025).
#'             A Simple Population Size Estimator for Local Minima Applied to Multidimensional Scaling.
#'
#'
#' @author Frank M.T.A. Busing
#' @export
#' @useDynLib fmds, .registration = TRUE

petersen <- function( first, second )
{
  s1 <- unique( first )
  s2 <- unique( second )
  marked <- length( s1 )
  drawn <- length( s2 )
  oldmarks <- sum( s2 %in% s1 )
  ifelse( oldmarks == 0, NA, marked * ( drawn / oldmarks ) )
} # petersen
