import { Box, Grid } from '@material-ui/core';
import React from 'react';

const pad = (num, size) => {
    var s = "000000000" + num;
    return s.substr(s.length-size);
}

const BlockDetails = ({data}) => {
    return (
        <Grid container>
            {data.map(({attributes}, index) => (
                <Grid item container style={{backgroundColor: 'lightgray', marginBottom: '10px', padding: '10px'}}>
                    <Grid item container style={{color: 'blue'}}>{pad(index+1, 3)}</Grid>
                    <Grid item container>{attributes.data}</Grid>
                </Grid>)
            )}
        </Grid>
    );
};

export default BlockDetails;