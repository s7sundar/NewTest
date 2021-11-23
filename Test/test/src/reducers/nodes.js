import {
  CHECK_NODE_STATUS_START, CHECK_NODE_STATUS_SUCCESS, CHECK_NODE_STATUS_FAILURE, 
  CHECK_BLOCK_SUCCESS, CHECK_BLOCK_FAILURE
} from '../constants/actionTypes';
import initialState from './initialState';

export default function nodesReducer(state = initialState().nodes, action) {
  let list, nodeIndex;
  switch (action.type) {
    case CHECK_NODE_STATUS_START:
      list = state.list;
      nodeIndex = state.list.findIndex(p => p.url === action.node.url);
      if (nodeIndex >= 0) {
        list = [
          ...state.list.slice(0, nodeIndex),
          {
            ...state.list[nodeIndex],
            loading: true
          },
          ...state.list.slice(nodeIndex + 1)
        ];
      }
      return {
        ...state,
        list
      };
    case CHECK_NODE_STATUS_SUCCESS:
      list = state.list;
      nodeIndex = state.list.findIndex(p => p.url === action.node.url);
      if (nodeIndex >= 0) {
        list = [
          ...state.list.slice(0, nodeIndex),
          {
            ...state.list[nodeIndex],
            online: true,
            name: action.res.node_name,
            loading: false
          },
          ...state.list.slice(nodeIndex + 1)
        ];
      }
      return {
        ...state,
        list
      };
    case CHECK_NODE_STATUS_FAILURE:
      list = state.list;
      nodeIndex = state.list.findIndex(p => p.url === action.node.url);
      if (nodeIndex >= 0) {
        list = [
          ...state.list.slice(0, nodeIndex),
          {
            ...state.list[nodeIndex],
            online: false,
            loading: false
          },
          ...state.list.slice(nodeIndex + 1)
        ];
      }
      return {
        ...state,
        list
      };
    case CHECK_BLOCK_SUCCESS:
      list = [...state.list];
      nodeIndex = state.list.findIndex(p => p.url === action.node.url);
      if (nodeIndex >= 0) {
        const nextListItem = {...list[nodeIndex]};
        nextListItem.blockData = action.res.data;
        list[nodeIndex] = nextListItem;
      }
      return {
        ...state,
        list
      };

    case CHECK_BLOCK_FAILURE:
      list = [...state.list];
      nodeIndex = state.list.findIndex(p => p.url === action.node.url);
      if (nodeIndex >= 0) {
        const nextListItem = {...list[nodeIndex]};
        nextListItem.blockData = [];
        list[nodeIndex] = nextListItem;
      }
      return {
        ...state,
        list
      };
     
    default:
      return state;
  }
}
