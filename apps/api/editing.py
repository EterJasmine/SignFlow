from fastapi import HTTPException


def save_result(ident, result, reason='human_edit'):
    """重算教程文案后提交新版本；各路由模块共用，避免与 main 循环导入。"""
    from packages.pipeline.composer import compose_step
    from .versions import commit_result

    if result.get('frames'):
        # 该结果可能来自进程内缓存并与其它请求共享，换外层字典而不是就地写入。
        result = {**result, 'tutorial_steps': [compose_step(action, result) for action in result['actions']]}
    try:
        return commit_result(ident, result, reason, result.get('result_revision', 0))
    except ValueError as error:
        raise HTTPException(409, str(error))
