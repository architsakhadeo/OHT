package convformat

import (
	"strconv"
	"strings"
)

func ListStr2Float(spl, sep string) []float64 {
	str := strings.Split(spl, sep)
	ary := make([]float64, len(str))
	for i := 0; i < len(str); i++ {
		ary[i], _ = strconv.ParseFloat(str[i], 64)
	}
	return ary
}