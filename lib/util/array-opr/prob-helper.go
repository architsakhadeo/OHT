package arrayOpr

func NormalizeProb(prob []float64) []float64 {
	sum := 0.0
	for i := 0; i < len(prob); i++ {
		sum += prob[i]// + math.Pow(10, -6)
	}
	pdf := make([]float64, len(prob))
	for i := 0; i < len(prob); i++ {
		//pdf[i] = (prob[i] + math.Pow(10, -6)) / sum
		pdf[i] = prob[i] / sum
	}
	return pdf
}
