import React from "react";
import { shallow } from "enzyme";
import Node from "./Node";
import BlockDetails from "./BlockDetails";
import { Typography } from "@material-ui/core";

describe('<Node />', () => {
    it('check block details exists', () => {
        const node = {
            url: "https://thawing-springs-53971.herokuapp.com",
            online: false,
            name: "Node 1",
            loading: false,
            blockData: [{
                attributes:{
                    data:'test'
                }
            }]
        }; 
        const expanded = true; 
        const toggleNodeExpanded = jest.fn();

        const props = {
            node,
            expanded,
            toggleNodeExpanded
        };
        const wrapper = shallow(<Node {...props}  />);
        expect(wrapper.find(BlockDetails).exists()).toBeTruthy()
    });

    it('check if block details not exists', () => {
        const node = {
            url: "https://thawing-springs-53971.herokuapp.com",
            online: false,
            name: "Node 1",
            loading: false,
            blockData: []
        }; 
        const expanded = true; 
        const toggleNodeExpanded = jest.fn();

        const props = {
            node,
            expanded,
            toggleNodeExpanded
        };
        const wrapper = shallow(<Node {...props}  />);
        expect(wrapper.find(Typography).at(2).text()).toEqual('No details found.');
    });
});