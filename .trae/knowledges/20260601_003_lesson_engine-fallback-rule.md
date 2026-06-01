# 回测引擎禁止回退必须并行验证

> **Type:** lesson
> **Date:** 2026-06-01
> **Context:** (unspecified)

## Summary

PyBroker失败时自研引擎不得回退应并行验证

## Background

_(none provided)_

## Details

当前PyBrokerBacktestRunner.run中PyBroker失败时静默回退到自研引擎。用户不知道PyBroker失败，掩盖bug。规则：1.禁止回退，PyBroker失败直接报错；2.自研引擎与PyBroker并行运行作验证；3.核心指标差异超10%发出警告；4.自研引擎仅用于交叉验证和边缘测试。

## Key Takeaways

- 自研引擎不做回退方案
- PyBroker失败必须报错
- 两套引擎并行运行对比验证
- 核心指标差异超10%警告

## Related

_(none)_
