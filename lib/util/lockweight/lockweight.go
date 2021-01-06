package lockweight

type LockWeight struct {
	UseLock       bool   `json:"lock-weight"`
	LockCondition string `json:"lock-condition"`

	DecCount int
	BestAvg  float64
	LockAvgRwd  float64 `json:"lock-condition-reward"`
	LockAvgLen  float64 `json:"lock-condition-length"`
	LockThrd int     `json:"lock-condition-thrd"`

	CheckChange LockFunc
}


type LockFunc func() bool
