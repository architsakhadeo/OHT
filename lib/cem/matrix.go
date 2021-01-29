package cem

import (
	"errors"
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// This file contains helper functions for matrix calculations.

func choleskySymmetricFromCovariance(covariance *mat.Dense, num int) (*mat.Cholesky, error) {
	covariance, err := nearestPD(covariance)
	if err != nil {
		return nil, err
	}

	symCov := mat.NewSymDense(num, nil)
	for i := 0; i < num; i++ {
		for j := 0; j < num; j++ {
			symCov.SetSym(i, j, (covariance.At(i, j)+covariance.At(j, i))/2.0)
		}
	}

	var choleskySymmetricCovariance mat.Cholesky
	choleskySymmetricCovariance.Factorize(symCov)
	return &choleskySymmetricCovariance, nil
}

func nearestPD(A *mat.Dense) (*mat.Dense, error) {
	ARows, AColumns := A.Dims()

	B := mat.NewDense(ARows, AColumns, nil)

	transposedA := transpose(A)

	for i := 0; i < ARows; i++ {
		for j := 0; j < AColumns; j++ {
			value := (A.At(i, j) + transposedA.At(i, j)) / 2.0
			B.Set(i, j, value)
		}
	}

	u, s, v, err := svd(B)
	if err != nil {
		return nil, err
	}

	uRows, uColumns := u.Dims()
	vRows, vColumns := v.Dims()
	uDense := mat.NewDense(uRows, uColumns, nil)
	vDense := mat.NewDense(vRows, vColumns, nil)

	for i := 0; i < uRows; i++ {
		for j := 0; j < uColumns; j++ {
			uDense.Set(i, j, u.At(i, j))
		}
	}

	for i := 0; i < vRows; i++ {
		for j := 0; j < vColumns; j++ {
			vDense.Set(i, j, v.At(i, j))
		}
	}

	diagonalSMatrix := mat.NewDense(ARows, AColumns, nil)
	for i := 0; i < ARows; i++ {
		for j := 0; j < AColumns; j++ {
			if i == j {
				diagonalSMatrix.Set(i, j, s[i])
			} else {
				diagonalSMatrix.Set(i, j, 0.0)
			}

		}
	}

	var temp mat.Dense
	var original mat.Dense
	temp.Mul(uDense, diagonalSMatrix)
	original.Mul(&temp, transpose(vDense))
	originalRows, originalColumns := original.Dims()
	originalDense := mat.NewDense(originalRows, originalColumns, nil)

	for i := 0; i < originalRows; i++ {
		for j := 0; j < originalColumns; j++ {
			originalDense.Set(i, j, original.At(i, j))
		}
	}

	var temp0 mat.Dense
	var H mat.Dense
	temp0.Mul(diagonalSMatrix, vDense)
	H.Mul(transpose(vDense), &temp0)
	hRows, hColumns := H.Dims()
	hDense := mat.NewDense(hRows, hColumns, nil)

	for i := 0; i < hRows; i++ {
		for j := 0; j < hColumns; j++ {
			hDense.Set(i, j, H.At(i, j))
		}
	}

	A2 := mat.NewDense(ARows, AColumns, nil)

	for i := 0; i < ARows; i++ {
		for j := 0; j < AColumns; j++ {
			A2.Set(i, j, (B.At(i, j)+hDense.At(i, j))/2.0)
		}
	}

	A3 := mat.NewDense(ARows, AColumns, nil)

	transposedA2 := transpose(A2)

	for i := 0; i < ARows; i++ {
		for j := 0; j < AColumns; j++ {
			A3.Set(i, j, (A2.At(i, j)+transposedA2.At(i, j))/2.0)
		}
	}

	if isPD(A3) {
		return A3, nil
	}

	normA := mat.Norm(A, 2)
	nextNearestDistance := math.Nextafter(normA, normA+1) - normA
	previousNearestDistance := normA - math.Nextafter(normA, normA-1)

	spacing := math.Min(nextNearestDistance, previousNearestDistance)

	I := mat.NewDense(ARows, ARows, nil)
	for i := 0; i < ARows; i++ {
		for j := 0; j < ARows; j++ {
			if i == j {
				I.Set(i, j, 1)
			} else {
				I.Set(i, j, 0)
			}
		}
	}

	k := 1.0

	for !isPD(A3) {
		var eig mat.Eigen
		ok := eig.Factorize(A3, mat.EigenNone)
		if !ok {
			return nil, errors.New("Eigen decomposition failed")
		}
		eigenvalues := eig.Values(nil)

		realeigenvalues := make([]float64, len(eigenvalues))
		for i := range eigenvalues {
			realeigenvalues[i] = real(eigenvalues[i])
		}

		minrealeigenvalues := realeigenvalues[0]
		for _, value := range realeigenvalues {
			if minrealeigenvalues > value {
				minrealeigenvalues = value
			}
		}
		for i := 0; i < ARows; i++ {
			for j := 0; j < AColumns; j++ {
				A3.Set(i, j, I.At(i, j)*((-minrealeigenvalues*math.Pow(k, 2))+spacing))
			}
		}
		k++
	}

	return A3, nil
}

func isPD(matrix mat.Matrix) bool {
	boolean := true

	var eig mat.Eigen
	ok := eig.Factorize(matrix, mat.EigenNone)
	if !ok {
		return false
	}
	eigenvalues := eig.Values(nil)

	realeigenvalues := make([]float64, len(eigenvalues))
	for i := range eigenvalues {
		realeigenvalues[i] = real(eigenvalues[i])
	}

	for i := range realeigenvalues {
		if realeigenvalues[i] <= 0 {
			boolean = false
			break
		}
	}

	return boolean
}

func transpose(matrix *mat.Dense) *mat.Dense {
	rows, columns := matrix.Dims()
	transposeMatrix := mat.NewDense(rows, columns, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			transposeMatrix.Set(i, j, matrix.At(j, i))
		}
	}
	return transposeMatrix
}

func svd(Matrix mat.Matrix) (mat.Dense, []float64, mat.Dense, error) {
	var svd mat.SVD
	if ok := svd.Factorize(Matrix, mat.SVDFull); !ok {
		var nilMat mat.Dense
		return nilMat, nil, nilMat, errors.New("SVD factorization failed")
	}
	var v, u mat.Dense
	svd.UTo(&u)
	svd.VTo(&v)
	s := svd.Values(nil)
	return u, s, v, nil
}

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}
