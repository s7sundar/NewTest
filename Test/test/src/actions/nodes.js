import fetch from "cross-fetch";
import * as types from "../constants/actionTypes";

const checkNodeStatusStart = (node) => {
  return {
    type: types.CHECK_NODE_STATUS_START,
    node,
  };
};

const checkNodeStatusSuccess = (node, res) => {
  return {
    type: types.CHECK_NODE_STATUS_SUCCESS,
    node,
    res,
  };
};

const checkNodeStatusFailure = (node) => {
  return {
    type: types.CHECK_NODE_STATUS_FAILURE,
    node,
  };
};


export function checkNodeStatus(node) {
  return async (dispatch) => {
    try {
      dispatch(checkNodeStatusStart(node));
      const res = await fetch(`${node.url}/api/v1/status`);

      if (res.status >= 400) {
        dispatch(checkNodeStatusFailure(node));
        return;
      }

      const json = await res.json();

      dispatch(checkNodeStatusSuccess(node, json));
    } catch (err) {
      dispatch(checkNodeStatusFailure(node));
    }
  };
}

const checkBlockStatusSuccess = (node, res) => {
  return {
    type: types.CHECK_BLOCK_SUCCESS,
    node,
    res,
  };
};

const checkBlockStatusFailure = (node) => {
  return {
    type: types.CHECK_BLOCK_FAILURE,
    node,
  };
};


export function checkBlockStatus(node) {
  return async (dispatch) => {
    try {
      const res = await fetch(`${node.url}/api/v1/blocks`);

      if (res.status >= 400) {
        dispatch(checkBlockStatusFailure(node));
        return;
      }

      const json = await res.json();

      dispatch(checkBlockStatusSuccess(node, json));
    } catch (err) {
      dispatch(checkBlockStatusFailure(node));
    }
  };
}

export function checkNodeStatuses(list) {
  return (dispatch) => {
    list.forEach((node) => {
      dispatch(checkNodeStatus(node));
      dispatch(checkBlockStatus(node));
    });
  };
}
