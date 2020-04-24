package experiment

// Experiment runs an experiment. The experiment itself is actually just a config file.
// Offline is managed by passing an extra value at the end of the State slice. Possibly add an API to Environment to say when it's in offline mode, but this seems specific to its implementation.
// Online off-policy can be handled by an OffPolicyEnvironment, which has instances of (behavior) Agent and Environment. It first passes state/reward to BehaviorAgent, which returns an Action. That action is appended to the State and passed out. The (target) Agent receives the state/reward and returns an Action (which had better be identical to the behavior agent). The Environment checks that the actions were the same, and throws an error if it wasn't (since that's a coding bug). Or the Environment ignores the Action if we want to allow that weird behavior.
